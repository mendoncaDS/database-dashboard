
import os
import pytz

import time as t
import pandas as pd
import streamlit as st
import plotly.express as px

from dotenv import load_dotenv
from datetime import datetime, time
from sqlalchemy import create_engine, text

# Environment setup
load_dotenv()
database_password = os.environ.get('password')
server_address = os.environ.get('serverip')
connection_string = f"postgresql://postgres:{database_password}@{server_address}:5432/postgres"
engine = create_engine(connection_string, echo=False)

# Constants
FREQUENCY_MAPPING = {
    "1 minute": "1T",
    "5 minutes": "5T",
    "15 minutes": "15T",
    "30 minutes": "30T",
    "1 hour": "1H",
    "2 hours": "2H",
    "4 hours": "4H",
    "6 hours": "6H",
    "12 hours": "12H",
    "1 day": "1D",
    "1 week": "1W",
    "1 month": "1M"
}

def get_human_readable_freq(internal_freq):
    for key, value in FREQUENCY_MAPPING.items():
        if value == internal_freq:
            return key
    return None  # In case no match is found

MIN_DATETIME = datetime(2017, 9, 1).replace(tzinfo=pytz.UTC)  # The oldest date from which we can query data

# Initialization
if 'dataframes_dict' not in st.session_state:
    st.session_state.dataframes_dict = {}
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = None
if 'selected_freq' not in st.session_state or st.session_state.selected_freq not in FREQUENCY_MAPPING.values():
    st.session_state.selected_freq = "1H"  # Default to 1 hour

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
    return data.resample(freq).apply(agg_dict)


def plot_in_placeholder(series, placeholder):

    if not isinstance(series, pd.Series):
        st.error("The input is not a Pandas Series.")
        return

    if not isinstance(series.index, pd.DatetimeIndex):
        st.error("The Series does not have a datetime index.")
        return

    # Convert the series to a dataframe for compatibility with plotly.express
    df = series.reset_index()
    df.columns = ["Date", "Value"]

    # Create interactive plot using plotly.express
    fig = px.line(df, x='Date', y='Value')
    
    # Check if logarithmic scale is selected
    if st.session_state.get('log_scale', False):
        fig.update_yaxes(type="log")

    # Update the layout to adjust the height and width
    fig.update_layout(autosize=True,height=650)  # You can adjust the width and height values as needed

    placeholder.plotly_chart(fig,use_container_width=True,theme="streamlit")


def symbol_selector():
    """
    Function to display only the symbol dropdown.
    """
    # Fetch unique symbols and sort them alphabetically
    unique_symbols = sorted(list(set([x[0] for x in unique_symbol_freqs()])))

    # Default to "BTCUSDT" if not already selected
    default_symbol_index = unique_symbols.index("BTCUSDT") if "BTCUSDT" in unique_symbols else 0

    
    
    selected_symbol = st.selectbox(
        "Select a Symbol",
        unique_symbols,
        index=default_symbol_index,
        key='selected_symbol_key'
    )

        

    # Update the session state with the selected symbol
    st.session_state.selected_symbol = selected_symbol



def frequency_selector():
    """
    Function to display the frequency dropdown outside the form.
    """
    selected_freq_display = st.selectbox("Select Frequency", list(FREQUENCY_MAPPING.keys()), index=list(FREQUENCY_MAPPING.keys()).index("1 hour"))
    # Update the session state with the actual resampling code
    st.session_state.selected_freq = FREQUENCY_MAPPING[selected_freq_display]



def symbol_freq_selector():
    """
    Function to display the symbol and frequency dropdowns.
    """
    # Fetch unique symbol and frequency combinations
    symbol_freq_combs = unique_symbol_freqs()

    # Extract unique symbols
    unique_symbols = list({x[0] for x in symbol_freq_combs})

    col1, col2 = st.columns(2)
    # Dropdowns for symbols and frequencies
    with col1:
        selected_symbol = st.selectbox(
            "Select a Symbol",
            unique_symbols,
            index=unique_symbols.index("BTCUSDT") if "BTCUSDT" in unique_symbols else 0,
            key='selected_symbol_key'
        )
    with col2:
        selected_freq_display = st.selectbox(
            "Select a Frequency",
            list(FREQUENCY_MAPPING.keys()),
            index=list(FREQUENCY_MAPPING.keys()).index("1 hour"),
            key='selected_freq_key'
        )
        # Update the session state with the actual resampling code
        st.session_state.selected_freq = FREQUENCY_MAPPING[selected_freq_display]

    # Update the session state with the selected values
    st.session_state.selected_symbol = selected_symbol


def datetime_selector():
    # Date and Hour Selector
    #col1, col2, col3 = st.columns(3)
    selected_date = st.date_input("Select a Date", datetime.now())
    selected_hour = st.slider("Select an Hour", 0, 23, 0) # Slider from 0 to 23 for hours
    selected_minute = st.slider("Select a Minute", 0, 59, 0) # Slider from 0 to 23 for hours
    selected_datetime = datetime.combine(selected_date, time(hour=selected_hour, minute=selected_minute))

    return selected_datetime


def get_last_timestamp(symbol, freq):
    with engine.begin() as connection:
        sql_query = "SELECT timestamp FROM market_data WHERE symbol = :symbol AND frequency = :freq ORDER BY timestamp DESC LIMIT 1;"
        result = connection.execute(text(sql_query),{"symbol":symbol,"freq":freq})
        last_timestamp = result.scalar()
    return last_timestamp


def load_symbol_data(symbol, start, end_datetime):
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


def unique_symbol_freqs():
    with engine.connect() as connection:
        query = text(f"SELECT DISTINCT symbol, frequency FROM market_data")
        result  = connection.execute(query)
        unique_combinations = result.fetchall()
    return unique_combinations


def main():
    st.set_page_config(
        page_title="Binance Dashboard",
        page_icon="graph-emoji.png",
        layout="wide",
    )

    # Initialize to 'Select version' by default
    if 'version' not in st.session_state:
        st.session_state.version = ''

    # Only show 'Choose a version' prompt on main page if version hasn't been selected or is 'Select version'
    if st.session_state.version == '':
        st.write("Are you on Desktop or Mobile?")
        version_options = ['', 'Desktop', 'Mobile']
        version = st.selectbox('Choose a version:', version_options, index=0)
        
        # If user selects a valid version, update the session state
        if version in ['Desktop', 'Mobile']:
            st.session_state.version = version
            st.experimental_rerun()  # Rerun the script to load the new content
        else:
            st.stop()

    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = None
    if 'selected_freq' not in st.session_state:
        st.session_state.selected_freq = None

    # Rest of your code
    if st.session_state.version == 'Desktop':
        pages = {
            "📈 Market Prices": prices_page_desktop,
            "🤖 Bots": bots_page,
        }

        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", list(pages.keys()))
        st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
        pages[page]()
    elif st.session_state.version == 'Mobile':
        pages = {
            "📈 Market Prices": prices_page_mobile,
            "🤖 Bots": bots_page,
        }

        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", list(pages.keys()))
        st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
        pages[page]()
        

def fetch_and_plot_data(graph_name_placeholder, graph_placeholder, selected_datetime, end_datetime, start_time):
    
    # Initialize most_recent_dt to None
    most_recent_dt = None
    
    # Check if the data is already available before querying the database
    if st.session_state.selected_symbol in st.session_state.dataframes_dict:
        most_recent_dt = st.session_state.dataframes_dict[st.session_state.selected_symbol].index.max()
        if most_recent_dt and most_recent_dt.tzinfo is None:
            most_recent_dt = most_recent_dt.tz_localize('UTC')
    else:
        most_recent_dt = get_last_timestamp(st.session_state.selected_symbol, "1m")
        if most_recent_dt and most_recent_dt.tzinfo is None:
            most_recent_dt = most_recent_dt.replace(tzinfo=pytz.UTC)
    
    if selected_datetime and selected_datetime > most_recent_dt:
        st.warning("Please select a date that's before the most recent data available.")
        return

    # If the selected date is older than MIN_DATETIME, adjust it to MIN_DATETIME
    if selected_datetime < MIN_DATETIME:
        selected_datetime = MIN_DATETIME

    # Check if the required data exists locally
    if st.session_state.selected_symbol in st.session_state.dataframes_dict:
        earliest_timestamp_local = st.session_state.dataframes_dict[st.session_state.selected_symbol].index.min()
        
        if earliest_timestamp_local <= selected_datetime:
            data = st.session_state.dataframes_dict[st.session_state.selected_symbol].loc[selected_datetime:end_datetime]
        else:
            data = load_symbol_data(st.session_state.selected_symbol, selected_datetime, end_datetime)

    else:
        data = load_symbol_data(st.session_state.selected_symbol, selected_datetime, end_datetime)

    # Processing and plotting the data
    if len(data) > 0:
        # Resample data to the selected frequency
        resampled_data = resample_data(data, st.session_state.selected_freq)
        if len(resampled_data) > 1000000:
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

    
def prices_page_desktop():
    st.title("📈 Binance Market Prices")

    # Create placeholder for the graph name early
    graph_name_placeholder = st.empty()

    # Placeholder for the graph
    graph_placeholder = st.empty()

    # If there's old data in session state, plot it before querying new data
    if hasattr(st.session_state, "graph_data") and st.session_state.graph_data is not None:
        plot_in_placeholder(st.session_state.graph_data["close"], graph_placeholder)

    st.sidebar.title("Data to fetch:")
    with st.sidebar.form(key='my_form'):
        # Symbol and Frequency selectors inside the form
        symbol_selector()
        frequency_selector()

        # Set default value for date selector to 6 months ago
        six_months_ago = datetime.today() - pd.DateOffset(months=6)
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.date_input("Select Start Date", six_months_ago)
        with col2:
            # End Time Selector
            end_date = st.date_input("Select End Date", datetime.now())

        # Convert the selected_date and end_date to datetime objects and localize to UTC
        selected_datetime = datetime.combine(selected_date, time(0, 0)).replace(tzinfo=pytz.UTC)
        end_datetime = datetime.combine(end_date, time(0, 0)).replace(tzinfo=pytz.UTC)

        # The submit button for the form
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button(label='Load Data')
        with col2:    
            st.session_state.log_scale = st.checkbox("Log Y", value=True, key='log_scale_key')

    # Fetch and plot the data using the selected start and end timestamps
    if submit_button or (hasattr(st.session_state, "selected_freq") and st.session_state.selected_freq != "1h"):
        start_time = t.time()  # Start timing here
        fetch_and_plot_data(graph_name_placeholder, graph_placeholder, selected_datetime, end_datetime, start_time)



    # Update the graph_name_placeholder with the current symbol and frequency after the form is processed
    if st.session_state.selected_symbol is not None:
        human_readable_freq = get_human_readable_freq(st.session_state.selected_freq)
        graph_name_placeholder.write(f"💲 {st.session_state.selected_symbol} - {human_readable_freq}")
    else:
        graph_name_placeholder.write("💲")

def prices_page_mobile():
    st.title("📈 Binance Market Prices")
    graph_name_placeholder = st.empty()
    graph_placeholder = st.empty()

    # Inject custom CSS to adjust column widths
    st.write('''<style>
    [data-testid="column"] {
        width: calc(33.3333% - 1rem) !important;
        flex: 1 1 calc(33.3333% - 1rem) !important;
        min-width: calc(33% - 1rem) !important;
    }
    </style>''', unsafe_allow_html=True)

    # If there's old data in session state, plot it before querying new data
    if hasattr(st.session_state, "graph_data") and st.session_state.graph_data is not None:
        plot_in_placeholder(st.session_state.graph_data["close"], graph_placeholder)

    # Move form to main area
    with st.form(key='my_form'):
        st.title("Data to fetch:")
        
        # Symbol and Frequency selectors inside the form
        symbol_selector()
        frequency_selector()

        # Set default value for date selector to 6 months ago
        six_months_ago = datetime.today() - pd.DateOffset(months=6)
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.date_input("Select Start Date", six_months_ago)
        with col2:
            # End Time Selector
            end_date = st.date_input("Select End Date", datetime.now())

        # Convert the selected_date and end_date to datetime objects and localize to UTC
        selected_datetime = datetime.combine(selected_date, time(0, 0)).replace(tzinfo=pytz.UTC)
        end_datetime = datetime.combine(end_date, time(0, 0)).replace(tzinfo=pytz.UTC)

        # Adjusting column widths to force side-by-side layout
        col1, col2 = st.columns([1,1])
        with col1:
            submit_button = st.form_submit_button(label='Load Data')
        with col2:
            st.session_state.log_scale = st.checkbox("Log Y", value=True, key='log_scale_key')

    # Check if the form was submitted
    if submit_button or (hasattr(st.session_state, "selected_freq") and st.session_state.selected_freq != "1h"):
        start_time = t.time()  # Start timing here
        fetch_and_plot_data(graph_name_placeholder, graph_placeholder, selected_datetime, end_datetime, start_time)
    elif hasattr(st.session_state, "graph_data") and st.session_state.graph_data is not None:
        plot_in_placeholder(st.session_state.graph_data["close"], graph_placeholder)

    # Update the graph_name_placeholder with the current symbol and frequency after the form is processed
    if st.session_state.selected_symbol is not None:
        human_readable_freq = get_human_readable_freq(st.session_state.selected_freq)
        graph_name_placeholder.write(f"💲 {st.session_state.selected_symbol} - {human_readable_freq}")
    else:
        graph_name_placeholder.write("💲")


def bots_page():
    st.title("🤖 Bots")
    st.subheader("Coming soon!")

if __name__ == "__main__":
    main()
