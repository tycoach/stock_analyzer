import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from statsmodels.tsa.seasonal import seasonal_decompose
import os
from dotenv import load_dotenv



load_dotenv()


# Configurations

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST =os.getenv("DB_HOST")
DB_PORT =os.getenv("DB_PORT")

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Custom CSS for enhanced styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
                min-height: 100vh;
                background-attachment: fixed;
            }
            .card {
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                border-radius: 12px;
                background-color: rgba(255, 255, 255, 0.95);
                padding: 25px;
                margin-bottom: 25px;
                backdrop-filter: blur(10px);
            }
            .main-title {
                color: #ff0000;
                font-weight: bold;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                font-size: 2.5em;
            }
            .section-title {
                color: #2c3e50;
                font-weight: 600;
                margin-bottom: 15px;
            }
            .dropdown {
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''



# Database connection
def get_db_connection():
    engine = create_engine(
    # Construct the connection string
        f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    )
    return engine


# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Real-Time Stock Analysis Dashboard", 
                        className="text-center mb-4 main-title")
            ], className="card")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Select Stock Symbol", className="section-title"),
                dcc.Dropdown(
                    id='symbol-selector',
                    placeholder='Select a stock symbol',
                    className="mb-4 dropdown",
                    style={
                        'backgroundColor': 'white',
                        'maxHeight': '200px',  # Sets maximum height for dropdown
                        'overflowY': 'auto'   # Enables vertical scrolling
                    }
                ),
                dcc.Interval(
                    id='interval-component',
                    interval=60*1000,
                    n_intervals=0
                )
            ], className="card", style={'marginBottom': '20px'})  # Adds spacing below this card
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Historical Price Analysis", className="section-title"),
                dcc.Graph(id='time-series-plot')
            ], className="card")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Forecast Analysis", className="section-title"),
                dcc.Graph(id='combined-forecast-plot')
            ], className="card")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Forecast Metrics", className="section-title"),
                html.Div(id='metrics-table')
            ], className="card")
        ])
    ])
], fluid=True, style={'padding': '20px'})


@app.callback(
    Output('symbol-selector', 'options'),
    Input('interval-component', 'n_intervals')
)
def update_symbols(n):
    engine = get_db_connection()
    query = "SELECT DISTINCT symbol FROM stock_data"
    df = pd.read_sql(query, engine)
    return [{'label': symbol, 'value': symbol} for symbol in df['symbol']]

@app.callback(
    [Output('time-series-plot', 'figure'),
     Output('combined-forecast-plot', 'figure'),
     Output('metrics-table', 'children')],
    [Input('symbol-selector', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_graphs(selected_symbol, n):
    if not selected_symbol:
        return {}, {}, []

    engine = get_db_connection()

    # Fetch historical data
    historical_query = f"""
        SELECT datetime, close 
        FROM stock_data 
        WHERE symbol = '{selected_symbol}'
        ORDER BY datetime
    """
    historical_data = pd.read_sql(historical_query, engine)

    # Fetch forecast data
    forecast_query = f"""
        SELECT ds, yhat, yhat_lower, yhat_upper 
        FROM forecast_data 
        WHERE symbol = '{selected_symbol}'
        ORDER BY ds DESC 
        LIMIT 24
    """
    forecast_data = pd.read_sql(forecast_query, engine)

    # Fetch metrics
    metrics_query = f"""
        SELECT rmse, mae, mape, timestamp 
        FROM forecast_metrics 
        WHERE symbol = '{selected_symbol}'
        ORDER BY timestamp DESC 
        LIMIT 1
    """
    metrics_data = pd.read_sql(metrics_query, engine)

    # Create time series plot
    time_series_fig = go.Figure()
    time_series_fig.add_trace(
        go.Scatter(
            x=historical_data['datetime'],
            y=historical_data['close'],
            name='Historical',
            line=dict(color='#00ff00', width=2)
        )
    )
    time_series_fig.update_layout(
        title=dict(
            text=f'{selected_symbol} Historical Price',
            font=dict(size=24, color='#2c3e50')
        ),
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        plot_bgcolor='rgba(255, 255, 255, 0.95)',
        paper_bgcolor='rgba(255, 255, 255, 0.95)',
        hovermode='x unified'
    )

    # Create forecast plot
    forecast_fig = go.Figure()
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat'],
            name='Forecast',
            line=dict(color='red')
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_upper'],
            name='Upper Bound',
            line=dict(color='gray', dash='dash')
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_lower'],
            name='Lower Bound',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        )
    )
    forecast_fig.update_layout(
        title=f'{selected_symbol} Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )

    # Create metrics table
    if not metrics_data.empty:
        metrics_table = dbc.Table.from_dataframe(
            metrics_data.round(4),
            striped=True,
            bordered=True,
            hover=True,
            className="mt-4",
            style={
                'backgroundColor': 'white',
                'fontSize': '1.1em',
                'textAlign': 'center'
            }
        )
    else:
        metrics_table = html.Div("No metrics available")

    return time_series_fig, forecast_fig, metrics_table


if __name__ == '__main__':
    app.run_server(debug=True)