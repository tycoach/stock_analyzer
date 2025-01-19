import requests
import json
import pandas as pd
import numpy as np
import os
import datetime
import time
import psycopg2
from psycopg2.extras import execute_values
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()


# Configurations
API_KEY = os.getenv('12_API_KEY')
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST =os.getenv("DB_HOST")
symbol = ['AAPL', 'TSLA', 'AMZN', 'GOOGL', 'MSFT']

# Database configuration

db_config = {
    "host": DB_HOST,
    "port": 5432,
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD
}

# Table name
table_name = "stock_data"

def fetch_api_data(api_url: str, params: dict = None) -> dict:
    """
    Fetch data from an API and return it as JSON.
    """
    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched data from API")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from API: {e}")
        raise Exception(f"Error fetching data from API: {e}")

def generate_api_url(symbols: list, api_key: str) -> str:
    """
    Generate the API URL for fetching stock data.
    """
    joined_symbols = ','.join(symbols)
    end_point = f"https://api.twelvedata.com/time_series?interval=2h&symbol={joined_symbols}&apikey={api_key}"
    logger.info(f"Generated API URL for symbols: {joined_symbols}")
    return end_point

def extract_stock_data(symbols: list) -> dict:
    """
    Extract stock data from TWELVE DATA API for a list of symbols.
    """
    all_data = {}
    url = generate_api_url(symbols, API_KEY)
    data = fetch_api_data(url)
    for symbol in symbols:
        all_data[symbol] = data.get(symbol, {})
        logger.info(f"Extracted data for symbol: {symbol}")
    return all_data

def transform_stock_data(stock_data: dict) -> dict:
    """
    Transform the stock data for each symbol into a pandas DataFrame.
    """
    transformed_data = {}
    for symbol, data in stock_data.items():
        if data['status'] == 'ok':
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])

            # Debug information
            logger.info(f"Symbol {symbol} - Total records before transformation: {len(df)}")
            logger.info(f"Symbol {symbol} - Unique timestamps: {len(df['datetime'].unique())}")

            # Check for duplicate timestamps
            duplicates = df[df.duplicated(subset=['datetime'], keep=False)]
            if not duplicates.empty:
                logger.warning(f"Found {len(duplicates)} duplicate timestamps for {symbol}")

            transformed_data[symbol] = df
            logger.info(f"Transformed data for symbol: {symbol}")
    return transformed_data

def add_symbol_column(transformed_data: dict) -> pd.DataFrame:
    """
    Add a 'symbol' column to each DataFrame and concatenate them into a single DataFrame.
    """
    all_data = []
    for symbol, df in transformed_data.items():
        df['symbol'] = symbol
        all_data.append(df)
    combined_df = pd.concat(all_data)
    logger.info(f"Combined data - Total records: {len(combined_df)}")
    return combined_df

def create_table(cursor, table_name: str):
    """
    Create the database table with a composite primary key.
    """
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        datetime TIMESTAMP,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume INTEGER,
        symbol TEXT,
        PRIMARY KEY (datetime, symbol)
    );
    """
    cursor.execute(create_table_query)
    logger.info(f"Created or verified table: {table_name}")

def push_to_postgres(data: pd.DataFrame, table_name: str, db_config: dict):
    """
    Pushes data to a PostgreSQL table, creating the table if it doesn't exist.
    """
    if data.empty:
        logger.error("DataFrame is empty. Nothing to push.")
        raise ValueError("DataFrame is empty. Nothing to push.")

    # Debug information for duplicate timestamps
    duplicate_dates = data[data.duplicated(subset=['datetime', 'symbol'], keep=False)]
    logger.info(f"Number of rows with duplicate timestamps and symbols: {len(duplicate_dates)}")
    if not duplicate_dates.empty:
        logger.info("Sample of duplicates:")
        logger.info(duplicate_dates.head())

    # Check unique combinations of datetime and symbol
    unique_combinations = len(data.groupby(['datetime', 'symbol']).size())
    logger.info(f"Total records: {len(data)}")
    logger.info(f"Unique datetime-symbol combinations: {unique_combinations}")

    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Create table with composite primary key
        create_table(cursor, table_name)

        # Prepare data for insertion
        columns = ["datetime", "open", "high", "low", "close", "volume", "symbol"]
        values = []
        for _, row in data.iterrows():
            values.append(tuple(row[col] for col in columns))

        # Insert data with UPSERT logic
        columns_str = ", ".join(columns)
        query = f"""
        INSERT INTO {table_name} ({columns_str})
        VALUES %s
        ON CONFLICT (datetime, symbol)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
        """

        execute_values(cursor, query, values)
        conn.commit()
        logger.info(f"Successfully inserted {len(values)} rows into {table_name}")

    except Exception as e:
        logger.error(f"Error while pushing data to PostgreSQL: {e}")
        raise

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def main():
    try:
        # Extract data from API
        logger.info("Starting data extraction from API...")
        stock_data = extract_stock_data(symbol)

        # Transform data
        logger.info("Transforming stock data...")
        transformed_data = transform_stock_data(stock_data)

        # Combine and process data
        logger.info("Combining data from all symbols...")
        combined_data = add_symbol_column(transformed_data)

        # Push to PostgreSQL
        logger.info(f"Pushing data to PostgreSQL... Total records to process: {len(combined_data)}")
        push_to_postgres(combined_data, table_name, db_config)

        logger.info("Data processing completed successfully")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()