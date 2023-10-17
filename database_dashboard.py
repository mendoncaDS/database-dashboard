
# External Library Imports
import os
import pytz

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from sqlalchemy import create_engine
from datetime import datetime, time, timedelta

# Local Module Imports
from render import get_pages
from constants import FREQUENCY_MAPPING
from auxiliary import unique_symbol_freqs

# ------------ Session State Initialization ------------

def initialize_session_state():
    if 'dataframes_dict' not in st.session_state:
        st.session_state.dataframes_dict = {}
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = "BTCUSDT"
    if 'selected_freq' not in st.session_state:
        st.session_state.selected_freq = "1H"  # Default to 1 hour
    if 'version' not in st.session_state:
        st.session_state.version = ''
    if 'engine' not in st.session_state:
            load_dotenv()
            database_password = os.environ.get('password')
            server_address = os.environ.get('serverip')
            connection_string = f"postgresql://postgres:{database_password}@{server_address}:5432/postgres"
            st.session_state.engine = create_engine(connection_string, echo=False)
    # Check if unique symbols are already in session_state
    if 'unique_symbols' not in st.session_state:
        # Fetch unique symbols from the database if they're not in session_state
        st.session_state.unique_symbols = sorted(list(set([x[0] for x in unique_symbol_freqs(st.session_state.engine)])))
    if 'start_datetime' not in st.session_state:
        default_start_datetime = datetime.today() - pd.DateOffset(months=6)
        st.session_state.start_datetime = datetime.combine(default_start_datetime, time(0, 0)).replace(tzinfo=pytz.UTC)
    if 'end_datetime' not in st.session_state:
        default_end_datetime = datetime.utcnow() + timedelta(days=1)
        st.session_state.end_datetime = datetime.combine(default_end_datetime, time(0, 0)).replace(tzinfo=pytz.UTC)
        st.session_state.end_datetime = datetime.utcnow() + timedelta(days=1)
    if 'log_scale' not in st.session_state:
        st.session_state.log_scale = True

# ------------ Main Function ------------

def main():
    initialize_session_state()

    st.set_page_config(
        page_title="Binance Dashboard",
        page_icon="graph-emoji.png",
        layout="wide",
    )

    # Only show 'Choose a version' prompt on main page if version hasn't been selected or is 'Select version'
    if st.session_state.version == '':
        st.write("Are you on Desktop or Mobile?")
        version = st.selectbox('Choose a version:', ['', 'Desktop', 'Mobile'], index=0)
        if version in ['Desktop', 'Mobile']:
            st.session_state.version = version
            st.experimental_rerun()
        else:
            st.stop()

    # Get the pages object
    pages = get_pages(st.session_state)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    pages[page](st.session_state.engine)

if __name__ == "__main__":
    main()
