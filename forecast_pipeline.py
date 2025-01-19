import pandas as pd
import numpy as np
from prophet import Prophet
from sqlalchemy import create_engine
import plotly.graph_objects as go
import time
import datetime as dt
import os
from dotenv import load_dotenv



load_dotenv()


# Configurations

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST =os.getenv("DB_HOST")
DB_PORT =os.getenv("DB_PORT")

# Database connection
def get_db_connection():
    engine = create_engine(
    # Construct the connection string
        f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    )
    return engine

# Fetch stock data
def get_stock_data(engine):
    query = """
        SELECT 
            datetime,
            symbol,
            close,
            COALESCE(high, close) AS high,
            COALESCE(low, close) AS low,
            COALESCE(volume, 0) AS volume
        FROM stock_data
        ORDER BY datetime;
    """
    try:
        with engine.connect() as conn:
            data = pd.read_sql(query, conn)
            data['datetime'] = pd.to_datetime(data['datetime'])
        return data
    except Exception as e:
        print(f"Error executing query: {e}")
        return None



# Analyze and forecast
def analyze_forecast(data, engine, forecast_horizon=24, data_interval="h"):
    results = {}
    try:
        if data is None or data.empty:
            print("No data available.")
            return results

        for stock in data['symbol'].unique():
            stock_subset = data[data['symbol'] == stock]

            # Prepare Prophet data
            prophet_data = stock_subset[['datetime', 'close']].rename(columns={'datetime': 'ds', 'close': 'y'})

            # Fit Prophet model
            model = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(prophet_data)

            # Generate forecast
            future_dates = model.make_future_dataframe(periods=forecast_horizon, freq=data_interval)
            forecast = model.predict(future_dates)

            # Calculate metrics
            actual = stock_subset['close'].values[-forecast_horizon:]
            predicted = forecast['yhat'][-forecast_horizon:]
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100

            metrics = pd.DataFrame({
                'symbol': [stock],
                'rmse': [rmse],
                'mae': [mae],
                'mape': [mape],
                'timestamp': [pd.Timestamp.now()]
            })

            # Store results
            results[stock] = {'forecast': forecast, 'metrics': metrics}

            # Save metrics to the database
            metrics.to_sql('forecast_metrics', engine, if_exists='append', index=False)

            # Save forecast data to the database
            forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_data.loc[:, 'symbol'] = stock
            forecast_data.loc[:, 'timestamp'] = pd.Timestamp.now()
            forecast_data.to_sql('forecast_data', engine, if_exists='append', index=False)

    except Exception as e:
        print(f"Error in analysis pipeline: {e}")

    return results



# Main execution
if __name__ == "__main__":
    engine = get_db_connection()
    # Fetch stock data
    stock_data = get_stock_data(engine)
    if stock_data is None or stock_data.empty:
        print("No stock data available to forecast.")
    else:
        # Run forecasting
        forecast_results = analyze_forecast(stock_data, engine)

        # Stop at forecast result
        print(forecast_results)
