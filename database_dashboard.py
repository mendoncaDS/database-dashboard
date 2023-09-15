
import os

import pandas as pd
import streamlit as st
import plotly.express as px

from dotenv import load_dotenv

from datetime import datetime, time
from sqlalchemy import create_engine, text



load_dotenv()
database_password = os.environ.get('password')
server_address = os.environ.get('serverip')

connection_string = f"postgresql://postgres:{database_password}@{server_address}:5432/postgres"
engine = create_engine(connection_string,echo=False)



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

    # Update the layout to adjust the height and width
    fig.update_layout(autosize=True,height=650)  # You can adjust the width and height values as needed

    placeholder.plotly_chart(fig,use_container_width=True,theme="streamlit")


def symbol_freq_selector():
    """
    Function to display the symbol and frequency dropdowns.
    """
    # Fetch unique symbol and frequency combinations
    symbol_freq_combs = unique_symbol_freqs()

    # Extract unique symbols and frequencies
    unique_symbols = list({x[0] for x in symbol_freq_combs})
    unique_freqs = list({x[1] for x in symbol_freq_combs})
    #st.write(f"Unique frequencies: {unique_freqs}")
    #print(f"Unique frequencies: {unique_freqs}")

    col1, col2 = st.columns(2)
    # Dropdowns for symbols and frequencies
    with col1:
        st.session_state.selected_symbol = st.selectbox(
            "Select a Symbol",
            [""] + unique_symbols,
            index=0 if not st.session_state.selected_symbol else unique_symbols.index(st.session_state.selected_symbol) + 1
        )
    with col2:
        st.session_state.selected_freq = st.selectbox(
            "Select a Frequency",
            [""] + unique_freqs,
            index=0 if not st.session_state.selected_freq else unique_freqs.index(st.session_state.selected_freq) + 1
        )


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


def is_combination_available(symbol, freq):
    """
    Check if a given symbol-frequency combination is available.
    """
    symbol_freq_combs = unique_symbol_freqs()
    return (symbol, freq) in symbol_freq_combs


def load_symbol_freq_data(symbol, freq, start):
    with engine.connect() as connection:
        values = {"symbol":symbol,"freq":freq, "start":start}
        query = text(f"SELECT timestamp, open, high, low, close, volume FROM market_data WHERE symbol = :symbol AND frequency = :freq AND timestamp >= :start")
        df = pd.read_sql_query(query, connection, params=values)
    
    df.set_index('timestamp', inplace=True)

    df.sort_index(inplace=True)

    df.index = df.index.tz_localize('UTC')

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
        


    
def prices_page_desktop():
    st.title("📈 Binance Market Prices")


    if st.session_state.selected_symbol is not None:
        graph_name_placeholder = st.empty()
        graph_name_placeholder.write(f"💲 {st.session_state.selected_symbol}")
    else:
        graph_name_placeholder = st.empty()
        graph_name_placeholder.write("💲")

    # Placeholder for the graph
    graph_placeholder = st.empty()

    # If there's old data in session state, plot it before querying new data
    if hasattr(st.session_state, "graph_data") and st.session_state.graph_data is not None:
        plot_in_placeholder(st.session_state.graph_data["close"], graph_placeholder)

    st.sidebar.title("Data to fetch:")
    # Move form to sidebar
    with st.sidebar.form(key='my_form'):
        # Symbol and Frequency selectors are inside the form
        symbol_freq_selector()

        selected_datetime = datetime_selector()

        # The submit button for the form
        submit_button = st.form_submit_button(label='Load Data')

    # Check if the form was submitted
    if submit_button:

        most_recent_dt = get_last_timestamp(st.session_state.selected_symbol, st.session_state.selected_freq)
        
        if not is_combination_available(st.session_state.selected_symbol, st.session_state.selected_freq):
            st.warning(f"The combination {st.session_state.selected_symbol} and {st.session_state.selected_freq} is not available.")
        elif selected_datetime and selected_datetime > most_recent_dt:
            st.warning("Please select a date that's before the most recent data available.")
        else:
            try:
                data = load_symbol_freq_data(
                    st.session_state.selected_symbol,
                    st.session_state.selected_freq,
                    selected_datetime
                )
                if len(data) > 0:
                    graph_name_placeholder.write(f"💲 {st.session_state.selected_symbol}")
                    st.session_state.graph_data = data  # Update the data in the session state
                    plot_in_placeholder(st.session_state.graph_data["close"], graph_placeholder)  # Re-plot with new data
                else:
                    st.warning("Please select an older start date.")
            except Exception as e:
                st.warning("Something went wrong.")
                st.error(f"PYTHON ERROR: {e}")

    elif hasattr(st.session_state, "graph_data") and st.session_state.graph_data is not None:
        # If the form hasn't been submitted in this rerun but there's stored data, plot the stored data
        plot_in_placeholder(st.session_state.graph_data["close"], graph_placeholder)

def prices_page_mobile():
    st.title("📈 Binance Market Prices")

    if st.session_state.selected_symbol is not None:
        graph_name_placeholder = st.empty()
        graph_name_placeholder.write(f"💲 {st.session_state.selected_symbol}")
    else:
        graph_name_placeholder = st.empty()
        graph_name_placeholder.write("💲")

    # Placeholder for the graph
    graph_placeholder = st.empty()

    # If there's old data in session state, plot it before querying new data
    if hasattr(st.session_state, "graph_data") and st.session_state.graph_data is not None:
        plot_in_placeholder(st.session_state.graph_data["close"], graph_placeholder)

    # Move form to sidebar
    with st.form(key='my_form'):
        st.title("Data to fetch:")
        # Symbol and Frequency selectors are inside the form
        symbol_freq_selector()

        selected_datetime = datetime_selector()

        # The submit button for the form
        submit_button = st.form_submit_button(label='Load Data')

    # Check if the form was submitted
    if submit_button:
        most_recent_dt = get_last_timestamp(st.session_state.selected_symbol, st.session_state.selected_freq)
        
        if not is_combination_available(st.session_state.selected_symbol, st.session_state.selected_freq):
            st.warning(f"The combination {st.session_state.selected_symbol} and {st.session_state.selected_freq} is not available.")
        elif selected_datetime and selected_datetime > most_recent_dt:
            st.warning("Please select a date that's before the most recent data available.")
        else:
            try:
                data = load_symbol_freq_data(
                    st.session_state.selected_symbol,
                    st.session_state.selected_freq,
                    selected_datetime
                )
                if len(data) > 0:
                    st.session_state.graph_data = data  # Update the data in the session state
                    plot_in_placeholder(st.session_state.graph_data["close"], graph_placeholder)  # Re-plot with new data
                else:
                    st.warning("Please select an older start date.")
            except Exception as e:
                st.warning("Something went wrong.")
                st.error(f"PYTHON ERROR: {e}")

    elif hasattr(st.session_state, "graph_data") and st.session_state.graph_data is not None:
        # If the form hasn't been submitted in this rerun but there's stored data, plot the stored data
        plot_in_placeholder(st.session_state.graph_data["close"], graph_placeholder)


def bots_page():
    st.title("🤖 Bots")
    st.subheader("Coming soon!")

if __name__ == "__main__":
    main()
