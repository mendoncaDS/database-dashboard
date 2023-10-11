
# External Library Imports
import pytz

import time as t
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

def load_symbol_data(symbol, start, end_datetime, engine):
    """
    Load data for the given symbol from the specified start datetime to the specified end datetime.
    Always fetches 1-minute granularity data.
    """
    start = max(start, MIN_DATETIME)  # Ensure the start date isn't before September 1st, 2017
    start = start.replace(tzinfo=pytz.UTC)
    end_datetime = end_datetime.replace(tzinfo=pytz.UTC)

    # Check if the data for this symbol is already in st.session_state.dataframes_dict
    if symbol in st.session_state.dataframes_dict:
        earliest_timestamp = st.session_state.dataframes_dict[symbol].index.min()
        latest_timestamp = st.session_state.dataframes_dict[symbol].index.max()
        
        # If start is earlier than the earliest timestamp we have or end is after the latest, fetch data for that timeframe
        if start < earliest_timestamp or end_datetime > latest_timestamp:
            with engine.connect() as connection:
                values = {"symbol": symbol, "start": start, "end": end_datetime}
                query = text(f"SELECT timestamp, open, high, low, close, volume FROM market_data WHERE symbol = :symbol AND frequency = '1m' AND timestamp >= :start AND timestamp <= :end")
                new_data = pd.read_sql_query(query, connection, params=values)

            new_data.set_index('timestamp', inplace=True)
            new_data.sort_index(inplace=True)
            new_data.index = new_data.index.tz_localize('UTC')

            # Merge the new data with the existing data
            df = pd.concat([new_data, st.session_state.dataframes_dict[symbol]])
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
        else:
            df = st.session_state.dataframes_dict[symbol]
    else:
        with engine.connect() as connection:
            values = {"symbol": symbol, "start": start, "end": end_datetime}
            query = text(f"SELECT timestamp, open, high, low, close, volume FROM market_data WHERE symbol = :symbol AND frequency = '1m' AND timestamp >= :start AND timestamp <= :end")
            df = pd.read_sql_query(query, connection, params=values)

        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df.index = df.index.tz_localize('UTC')

        # Save the loaded dataframe to st.session_state.dataframes_dict for future use
        st.session_state.dataframes_dict[symbol] = df

    # Slice the dataframe using the start and end timestamps
    df = df.loc[start:end_datetime]
    
    return df

def unique_symbol_freqs(engine):
    with engine.connect() as connection:
        query = text("SELECT DISTINCT symbol, frequency FROM market_data")
        result = connection.execute(query)
        unique_combinations = result.fetchall()
    return unique_combinations

# ---------------------- Data Processing Functions ----------------------

def process_indicators(graph_data, indicators_list):
    """ Process the indicators list and return a dataframe with the required columns. """
    df = graph_data.copy()  # Copy the entire dataframe initially
    counter = {"SMA": 0, "EMA": 0, "WMA": 0, "HMA": 0, "DEMA": 0, "TEMA": 0, "TRIMA": 0, "KAMA": 0, "ZLMA": 0, "ALMA": 0, "BBANDS": 0}
    for idx, indicator in enumerate(indicators_list):
        
        counter[indicator['indicator']] += 1  # Increment the counter for the current indicator type

        if indicator["indicator"] == "SMA":
            df[f"SMA_{counter['SMA']}"] = ta.sma(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "EMA":
            df[f"EMA_{counter['EMA']}"] = ta.ema(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "WMA":
            df[f"WMA_{counter['WMA']}"] = ta.wma(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "HMA":
            df[f"HMA_{counter['HMA']}"] = ta.hma(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "DEMA":
            df[f"DEMA_{counter['DEMA']}"] = ta.dema(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "TEMA":
            df[f"TEMA_{counter['TEMA']}"] = ta.tema(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "TRIMA":
            df[f"TRIMA_{counter['TRIMA']}"] = ta.trima(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "KAMA":
            df[f"KAMA_{counter['KAMA']}"] = ta.kama(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "ZLMA":
            df[f"ZLMA_{counter['ZLMA']}"] = ta.zlma(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "ALMA":
            df[f"ALMA_{counter['ALMA']}"] = ta.alma(df["close"], length=indicator['params']['period'])
        elif indicator["indicator"] == "BBANDS":
            bband = ta.bbands(df["close"], length=indicator['params']['length'], std=indicator['params']['std'])
            bands = [f"BBM_{counter['BBANDS']}", f"BBU_{counter['BBANDS']}", f"BBL_{counter['BBANDS']}"]
            for i, band in enumerate(bands):
                df[band] = bband.iloc[:, i]
    
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

def fetch_and_plot_data(graph_name_placeholder, graph_placeholder, selected_datetime, end_datetime, start_time, engine):
    
    # Initialize most_recent_dt to None
    most_recent_dt = None
    
    # Check if the data is already available before querying the database
    if st.session_state.selected_symbol in st.session_state.dataframes_dict:
        most_recent_dt = st.session_state.dataframes_dict[st.session_state.selected_symbol].index.max()
        if most_recent_dt and most_recent_dt.tzinfo is None:
            most_recent_dt = most_recent_dt.tz_localize('UTC')
    else:
        most_recent_dt = get_last_timestamp(st.session_state.selected_symbol, "1m", engine)
        if most_recent_dt and most_recent_dt.tzinfo is None:
            most_recent_dt = most_recent_dt.replace(tzinfo=pytz.UTC)
    
    if selected_datetime and selected_datetime > most_recent_dt:
        st.warning("Please select a date that's before the most recent data available.")
        return

    # If the most recent data in the database is newer than the most recent data we have locally, fetch it
    if most_recent_dt and (st.session_state.selected_symbol not in st.session_state.dataframes_dict or most_recent_dt > st.session_state.dataframes_dict[st.session_state.selected_symbol].index.max()):
        new_data = load_symbol_data(st.session_state.selected_symbol, most_recent_dt, end_datetime, engine)
        # If data is already present for this symbol, append the new data
        if st.session_state.selected_symbol in st.session_state.dataframes_dict:
            st.session_state.dataframes_dict[st.session_state.selected_symbol] = pd.concat([st.session_state.dataframes_dict[st.session_state.selected_symbol], new_data])
        else:
            st.session_state.dataframes_dict[st.session_state.selected_symbol] = new_data


    # If the selected date is older than MIN_DATETIME, adjust it to MIN_DATETIME
    if selected_datetime < MIN_DATETIME:
        selected_datetime = MIN_DATETIME

    # Check if the required data exists locally
    if st.session_state.selected_symbol in st.session_state.dataframes_dict:
        earliest_timestamp_local = st.session_state.dataframes_dict[st.session_state.selected_symbol].index.min()
        
        if earliest_timestamp_local <= selected_datetime:
            data = st.session_state.dataframes_dict[st.session_state.selected_symbol].loc[selected_datetime:end_datetime]
        else:
            data = load_symbol_data(st.session_state.selected_symbol, selected_datetime, end_datetime, engine)

    else:
        data = load_symbol_data(st.session_state.selected_symbol, selected_datetime, end_datetime, engine)
    

    # Processing and plotting the data
    if len(data) > 0:
        # Resample data to the selected frequency
        resampled_data = resample_data(data, st.session_state.selected_freq)
        if len(resampled_data) > 500000:
            st.warning("The data is too big for plotting. Please select a narrower time period or a broader time frame.")
            return
        
        graph_name_placeholder.write(f"💲 {st.session_state.selected_symbol} - {st.session_state.selected_freq}")
        st.session_state.graph_data = resampled_data  # Update the data in the session state
        plot_in_placeholder(st.session_state.graph_data["close"], graph_placeholder)  # Re-plot with new data
    else:
        st.warning("Please select an older start date.")

    # Mark form as submitted
    st.session_state.form_submitted = True

    # Calculate and display the time taken after plotting
    end_time = t.time()
    st.write(f"Plot loaded in {end_time - start_time:.2f} seconds.")

def plot_in_placeholder(data, placeholder):
    # Convert to DataFrame if it's a Series
    if isinstance(data, pd.Series):
        data = data.to_frame(name="Value")

    if not isinstance(data, pd.DataFrame):
        st.error("The input is not a Pandas DataFrame.")
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
