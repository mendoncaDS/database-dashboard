# External Library Imports
import pytz

import time as t
import pandas as pd
import streamlit as st

from datetime import datetime, timedelta, time

# Local Module Imports
from constants import FREQUENCY_MAPPING
from auxiliary import get_human_readable_freq, fetch_and_plot_data, unique_symbol_freqs, plot_in_placeholder

# --------------- Page Rendering Functions ---------------


def get_pages(session_state):
    
    def prices_page_desktop(engine):
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
            symbol_selector(engine, session_state)
            frequency_selector()

            # Set default value for date selector to 6 months ago
            six_months_ago = datetime.today() - pd.DateOffset(months=6)
            
            col1, col2 = st.columns(2)
            with col1:
                selected_date = st.date_input("Select Start Date", six_months_ago)
            with col2:
                # End Time Selector
                end_date_default = datetime.utcnow() + timedelta(days=1)
                end_date = st.date_input("Select End Date", end_date_default)

            # Convert the selected_date and end_date to datetime objects and localize to UTC
            selected_datetime = datetime.combine(selected_date, time(0, 0)).replace(tzinfo=pytz.UTC)
            current_time = datetime.utcnow().time()
            end_datetime = datetime.combine(end_date, current_time).replace(tzinfo=pytz.UTC)

            # The submit button for the form
            col1, col2 = st.columns(2)
            with col1:
                submit_button = st.form_submit_button(label='Load Data')
            with col2:
                st.session_state.log_scale = st.checkbox("Log Y", value=True, key='log_scale_key')

        # Fetch and plot the data using the selected start and end timestamps
        if submit_button or (hasattr(st.session_state, "selected_freq") and st.session_state.selected_freq != "1h"):
            start_time = t.time()  # Start timing here
            fetch_and_plot_data(graph_name_placeholder, graph_placeholder, selected_datetime, end_datetime, start_time, engine)

        # Update the graph_name_placeholder with the current symbol and frequency after the form is processed
        if st.session_state.selected_symbol is not None:
            human_readable_freq = get_human_readable_freq(st.session_state.selected_freq)
            graph_name_placeholder.write(f"💲 {st.session_state.selected_symbol} - {human_readable_freq}")
        else:
            graph_name_placeholder.write("💲")

    def prices_page_mobile(engine):
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
            symbol_selector(engine, session_state)
            frequency_selector()

            # Set default value for date selector to 6 months ago
            six_months_ago = datetime.today() - pd.DateOffset(months=6)
            
            col1, col2 = st.columns(2)
            with col1:
                selected_date = st.date_input("Select Start Date", six_months_ago)
            with col2:
                # End Time Selector
                end_date_default = datetime.utcnow() + timedelta(days=1)
                end_date = st.date_input("Select End Date", end_date_default)

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
            fetch_and_plot_data(graph_name_placeholder, graph_placeholder, selected_datetime, end_datetime, start_time, engine)
        elif hasattr(st.session_state, "graph_data") and st.session_state.graph_data is not None:
            plot_in_placeholder(st.session_state.graph_data["close"], graph_placeholder)

        # Update the graph_name_placeholder with the current symbol and frequency after the form is processed
        if st.session_state.selected_symbol is not None:
            human_readable_freq = get_human_readable_freq(st.session_state.selected_freq)
            graph_name_placeholder.write(f"💲 {st.session_state.selected_symbol} - {human_readable_freq}")
        else:
            graph_name_placeholder.write("💲")

    def indicators_page(engine):
        st.title("📊 Indicators")
        st.subheader("Coming soon!")

    def bots_page(engine):
        st.title("🤖 Bots")
        st.subheader("Coming soon!")

        
    return {
        "📈 Market Prices": (prices_page_desktop if session_state.version == 'Desktop' else prices_page_mobile),
        "📊 Indicators": indicators_page,
        "🤖 Bots": bots_page,
    }

# -------------- Widget Rendering Functions --------------

def symbol_selector(engine, session_state):
    """
    Function to display only the symbol dropdown.
    """
    # Check if unique symbols are already in session_state
    if 'unique_symbols' not in session_state:
        # Fetch unique symbols from the database if they're not in session_state
        session_state.unique_symbols = sorted(list(set([x[0] for x in unique_symbol_freqs(engine)])))
    
    # Default to "BTCUSDT" if not already selected
    default_symbol_index = session_state.unique_symbols.index("BTCUSDT") if "BTCUSDT" in session_state.unique_symbols else 0

    selected_symbol = st.selectbox(
        "Select a Symbol",
        session_state.unique_symbols,
        index=default_symbol_index,
        key='selected_symbol_key'
    )

    # Update the session state with the selected symbol
    session_state.selected_symbol = selected_symbol

def frequency_selector():
    """
    Function to display the frequency dropdown outside the form.
    """
    selected_freq_display = st.selectbox("Select Frequency", list(FREQUENCY_MAPPING.keys()), index=list(FREQUENCY_MAPPING.keys()).index("1 hour"))
    # Update the session state with the actual resampling code
    st.session_state.selected_freq = FREQUENCY_MAPPING[selected_freq_display]

def datetime_selector():
    # Date and Hour Selector
    selected_date = st.date_input("Select a Date", datetime.now())
    selected_hour = st.slider("Select an Hour", 0, 23, 0) # Slider from 0 to 23 for hours
    selected_minute = st.slider("Select a Minute", 0, 59, 0) # Slider from 0 to 23 for hours
    selected_datetime = datetime.combine(selected_date, time(hour=selected_hour, minute=selected_minute))
    return selected_datetime

