
# External Library Imports
import pytz

import pandas as pd
import pandas_ta as ta
import streamlit as st
import plotly.express as px

from sqlalchemy import text

# Local Module Imports
from constants import MIN_DATETIME, FREQUENCY_MAPPING

# ---------------------- Database Utility Functions ----------------------

def get_last_timestamp(symbol, freq, engine):
    with engine.begin() as connection:
        sql_query = "SELECT timestamp FROM market_data WHERE symbol = :symbol AND frequency = :freq ORDER BY timestamp DESC LIMIT 1;"
        result = connection.execute(text(sql_query),{"symbol":symbol,"freq":freq})
        last_timestamp = result.scalar()
    return last_timestamp


def load_symbol_data(symbol, start_datetime, end_datetime, engine):
    """
    Load data for the given symbol from the specified start datetime to the specified end datetime.
    Always fetches 1-minute granularity data.
    """
    start_datetime = max(start_datetime, MIN_DATETIME)  # Ensure the start date isn't before September 1st, 2017
    start_datetime = start_datetime.replace(tzinfo=pytz.UTC)
    end_datetime = end_datetime.replace(tzinfo=pytz.UTC)

    print(f"Loading data for {symbol} from {start_datetime} to {end_datetime}")

    with engine.connect() as connection:
        values = {"symbol": symbol, "start_datetime": start_datetime, "end": end_datetime}
        query = text(f"SELECT timestamp, open, high, low, close, volume FROM market_data WHERE symbol = :symbol AND frequency = '1m' AND timestamp >= :start_datetime AND timestamp <= :end")
        df = pd.read_sql_query(query, connection, params=values)

        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df.index = df.index.tz_localize('UTC')

        # Save the loaded dataframe to st.session_state.dataframes_dict for future use
        st.session_state.dataframes_dict[symbol] = df

    # Slice the dataframe using the start and end timestamps
    df = df.loc[start_datetime:end_datetime]

    start_date = df.index.min()
    end_date = df.index.max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq="1T")  # 'D' for daily frequency
    df = df.reindex(all_dates, method='ffill')
    
    return df


def unique_symbol_freqs(engine):
    with engine.connect() as connection:
        query = text("SELECT DISTINCT symbol, frequency FROM market_data;")
        result = connection.execute(query)
        unique_combinations = result.fetchall()
    return unique_combinations

def unique_bots(engine):
    with engine.connect() as connection:
        query = text("SELECT array_agg(DISTINCT bot_name) as bot_names FROM bots;")
        result = connection.execute(query)
        unique = result.scalar()
    return unique



def update_bot_data(engine, bot_name):
    """
    Check the local data for the specified bot, compare with the database,
    and update the local data if newer records are found in the database.

    Parameters:
    engine (sqlalchemy.engine.Engine): The database engine.
    bot_name (str): The name of the bot to check the data for.
    """

    # Check if there is local data for the bot
    local_data = st.session_state['bots_data'].get(bot_name)

    if local_data is not None and not local_data.empty:
        # If local data exists, find the most recent timestamp
        latest_local_timestamp = local_data['timestamp'].max()
        
        # Query the database for the most recent timestamp for this bot
        with engine.connect() as connection:
            query = text("SELECT MAX(timestamp) AS latest_timestamp FROM bots WHERE bot_name = :bot_name")
            result = connection.execute(query, {"bot_name": bot_name}).fetchone()
            latest_db_timestamp = result[0]

        # Compare the timestamps
        if latest_db_timestamp > latest_local_timestamp:
            # If the database has newer records, retrieve them
            with engine.connect() as connection:
                query = text("""
                    SELECT timestamp, position, portfolio_value, asset_value, symbol FROM bots 
                    WHERE bot_name = :bot_name AND timestamp > :latest_local_timestamp
                """)
                new_records = pd.read_sql(query, connection, params={"bot_name": bot_name, "latest_local_timestamp": latest_local_timestamp})

            # Append new records to the local data and update the session state
            updated_data = local_data.append(new_records, ignore_index=True)
            st.session_state['bots_data'][bot_name] = updated_data
    else:
        # If no local data exists, retrieve all data for this bot from the database
        with engine.connect() as connection:
            query = text("SELECT timestamp, position, portfolio_value, asset_value, symbol FROM bots WHERE bot_name = :bot_name")
            all_records = pd.read_sql(query, connection, params={"bot_name": bot_name})

        # Save the data to the session state
        st.session_state['bots_data'][bot_name] = all_records

    # For debugging purposes, you might want to return the data and see it
    return st.session_state['bots_data'][bot_name]


# ---------------------- Data Processing Functions ----------------------

def process_indicators(graph_data, indicators_list):
    """ Process the indicators list and return a dataframe with the required columns. """
    df = graph_data.copy()  # Copy the entire dataframe initially
    df.ta.strategy(indicators_list)
    counter = {"SMA": 0, "EMA": 0, "WMA": 0, "HMA": 0, "DEMA": 0, "TEMA": 0, "TRIMA": 0, "KAMA": 0, "ZLMA": 0, "ALMA": 0, "BBANDS": 0, "RSI": 0, "STOCH": 0, "MACD": 0}
    
    # Helper function to classify indicator types
    def classify_indicator(indicator_name):
        # Oscillators
        oscillators = ["RSI", "STOCH", "MACD"]  # Add other oscillator indicators as needed
        if indicator_name in oscillators:
            return "osc_"
        
        # By default, classify others as trend indicators
        return "trend_"
    
    for idx, indicator in enumerate(indicators_list):
        counter[indicator['indicator']] += 1  # Increment the counter for the current indicator type
        prefix = classify_indicator(indicator['indicator'])  # Classify the indicator

        # Now, adjust your naming convention to include the prefix for classification
        if indicator["indicator"] == "SMA":
            df[f"{prefix}SMA_{counter['SMA']}"] = ta.sma(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "EMA":
            df[f"{prefix}EMA_{counter['EMA']}"] = ta.ema(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "WMA":
            df[f"{prefix}WMA_{counter['WMA']}"] = ta.wma(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "HMA":
            df[f"{prefix}HMA_{counter['HMA']}"] = ta.hma(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "DEMA":
            df[f"{prefix}DEMA_{counter['DEMA']}"] = ta.dema(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "TEMA":
            df[f"{prefix}TEMA_{counter['TEMA']}"] = ta.tema(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "TRIMA":
            df[f"{prefix}TRIMA_{counter['TRIMA']}"] = ta.trima(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "KAMA":
            df[f"{prefix}KAMA_{counter['KAMA']}"] = ta.kama(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "ZLMA":
            df[f"{prefix}ZLMA_{counter['ZLMA']}"] = ta.zlma(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "ALMA":
            df[f"{prefix}ALMA_{counter['ALMA']}"] = ta.alma(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "BBANDS":
            bband = ta.bbands(df["close"], length=indicator['params']['length'], std=indicator['params']['std'])
            bands = [f"{prefix}BBM_{counter['BBANDS']}", f"{prefix}BBU_{counter['BBANDS']}", f"{prefix}BBL_{counter['BBANDS']}"]
            for i, band in enumerate(bands):
                df[band] = bband.iloc[:, i]
        elif indicator["indicator"] == "RSI":
            df[f"{prefix}RSI_{counter['RSI']}"] = ta.rsi(df["close"], length=indicator['params']['period'])

        elif indicator["indicator"] == "STOCH":
            stoch = ta.stoch(df["high"], df["low"], df["close"], 
                             fast_k_period=indicator['params']['fast_k_period'], 
                             slow_k_period=indicator['params']['slow_k_period'], 
                             slow_d_period=indicator['params']['slow_d_period'])
            stoch_keys = [f"{prefix}Stoch_k_{counter['STOCH']}", f"{prefix}Stoch_d_{counter['STOCH']}"]
            for i, key in enumerate(stoch_keys):
                df[key] = stoch.iloc[:, i]

        elif indicator["indicator"] == "MACD":
            macd = ta.macd(df["close"], 
                           fast_period=indicator['params']['fast_period'], 
                           slow_period=indicator['params']['slow_period'], 
                           signal_period=indicator['params']['signal_period'])
            
            macd_keys = [f"{prefix}MACD_{counter['MACD']}", f"{prefix}MACD_signal_{counter['MACD']}", f"{prefix}MACD_hist_{counter['MACD']}"]
            for i, key in enumerate(macd_keys):
                df[key] = macd.iloc[:, i]
    
    columns_to_keep = [col for col in df.columns if col not in ['open', 'high', 'low', 'volume']]
    return df[columns_to_keep]

def get_human_readable_freq(internal_freq):
    for key, value in FREQUENCY_MAPPING.items():
        if value == internal_freq:
            return key
    return None  # In case no match is found

def resample_data(data, freq):
    """
    Resamples the given data to the specified frequency using aggregation methods.
    """
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    resampled = data.resample(freq).apply(agg_dict)
    
    # Remove the last entry to ensure only complete candles are displayed for resampled data.
    # For 1m data, no need to trim as it's already done by the updater.
    if freq != "1T":
        return resampled.iloc[:-1]
    else:
        return resampled

def fetch_and_plot_data(graph_placeholder, selected_datetime, end_datetime, engine):
    # Ensure the selected_datetime is timezone-aware and in UTC
    if selected_datetime.tzinfo is None or selected_datetime.tzinfo.utcoffset(selected_datetime) is not None:
        selected_datetime = selected_datetime.replace(tzinfo=pytz.UTC)

    # Ensure the end_datetime is timezone-aware and in UTC
    if end_datetime.tzinfo is None or end_datetime.tzinfo.utcoffset(end_datetime) is not None:
        end_datetime = end_datetime.replace(tzinfo=pytz.UTC)

    symbol = st.session_state.selected_symbol

    # Fetch the last timestamp from the database for this symbol
    last_available_timestamp = get_last_timestamp(symbol, "1m", engine)

    # If there's no local data for this symbol, initialize it by fetching from the database
    if symbol not in st.session_state.dataframes_dict:
        st.session_state.dataframes_dict[symbol] = load_symbol_data(symbol, selected_datetime, end_datetime, engine)
    else:
        local_data = st.session_state.dataframes_dict[symbol]
        local_min_ts = local_data.index.min()
        local_max_ts = local_data.index.max()

        # If the local dataset doesn't reach back to the selected start date, fetch older data
        if selected_datetime < local_min_ts:
            older_data = load_symbol_data(symbol, selected_datetime, local_min_ts, engine)
            older_data = older_data.iloc[:-1]
            st.session_state.dataframes_dict[symbol] = pd.concat([older_data, local_data])

        # Ensure last_available_timestamp is timezone-aware by localizing it to UTC
        if last_available_timestamp.tzinfo is None:
            last_available_timestamp = pytz.UTC.localize(last_available_timestamp)

        # Ensure local_max_ts is timezone-aware by localizing it to UTC
        local_data = st.session_state.dataframes_dict[symbol]
        local_min_ts = local_data.index.min()
        if local_min_ts.tzinfo is None:
            local_min_ts = pytz.UTC.localize(local_min_ts)

        local_max_ts = local_data.index.max()
        if local_max_ts.tzinfo is None:
            local_max_ts = pytz.UTC.localize(local_max_ts)

        # If the local dataset doesn't extend to the most recent data available, fetch newer data
        if last_available_timestamp > local_max_ts:
            newer_data = load_symbol_data(symbol, local_max_ts, last_available_timestamp, engine)
            newer_data = newer_data.iloc[1:]
            st.session_state.dataframes_dict[symbol] = pd.concat([local_data, newer_data])

    # At this point, local data should span the required timeframe. Extract the necessary slice.
    data = st.session_state.dataframes_dict[symbol].loc[selected_datetime:end_datetime]
    

    # Processing and plotting the data
    if len(data) > 0:
        # Resample data to the selected frequency
        resampled_data = resample_data(data, st.session_state.selected_freq)
        if len(resampled_data) > 500000:
            st.warning("The data is too big for plotting. Please select a narrower time period or a broader time frame.")
            return
        
        st.session_state.graph_data = resampled_data  # Update the data in the session state
        plot_in_placeholder(st.session_state.graph_data["close"], graph_placeholder)  # Re-plot with new data
    else:
        st.warning("Please select an older start date.")


def plot_in_placeholder(data, placeholder):
    # Convert to DataFrame if it's a Series
    if isinstance(data, pd.Series):
        data = data.to_frame()

    if not isinstance(data, pd.DataFrame):
        st.error("The input is not a Pandas DataFrame or Series.")
        return

    if not isinstance(data.index, pd.DatetimeIndex):
        st.error("The DataFrame does not have a datetime index.")
        return

    # Create interactive plot using plotly.express
    fig = px.line(data)

    # Check if logarithmic scale is selected
    if st.session_state.get('log_scale', False):
        fig.update_yaxes(type="log")

    # Update the layout to adjust the height and width
    fig.update_layout(autosize=True, height=650)  # You can adjust the width and height values as needed

    placeholder.plotly_chart(fig, use_container_width=True, theme="streamlit")
