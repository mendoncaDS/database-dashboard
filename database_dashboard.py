
# External Library Imports
import os
import json
import pytz
import requests

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
    default_start_datetime = datetime.today() - pd.DateOffset(months=6)
    default_end_datetime = datetime.utcnow() + timedelta(days=1)
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
            github_token = os.environ.get('githubtoken')
            connection_string = f"postgresql://postgres:{database_password}@{server_address}:5432/postgres"
            st.session_state.engine = create_engine(connection_string, echo=False)

            headers = {
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3.raw'
            }

            zurfer_bots_config_url = 'https://api.github.com/repos/mendoncaDS/zlema_bots_config/contents/bots_config.json?ref=main'

            zurfer_bots_response = requests.get(zurfer_bots_config_url, headers=headers)
            if zurfer_bots_response.ok:
                bots_data_list = json.loads(zurfer_bots_response.text)
                st.session_state.bots_data_dict = {}
                # Loop through each dictionary in the list
                for bot in bots_data_list:
                    # Use the 'bot_name' as the key for the new dictionary
                    bot_name = bot.pop('bot_name')  # Remove 'bot_name' and get its value
                    st.session_state.bots_data_dict[bot_name] = bot  # Assign the remaining dictionary to the bot_name key
                st.session_state.unique_bots_list = [key for key in st.session_state.bots_data_dict.keys()]
            else:
                # Handle errors here
                print('Could not fetch the file:', zurfer_bots_response.status_code)


    # Check if unique symbols are already in session_state
    if 'unique_symbols' not in st.session_state:
        # Fetch unique symbols from the database if they're not in session_state
        st.session_state.unique_symbols = sorted(list(set([x[0] for x in unique_symbol_freqs(st.session_state.engine)])))
    if 'start_datetime' not in st.session_state:
        st.session_state.start_datetime = datetime.combine(default_start_datetime, time(0, 0)).replace(tzinfo=pytz.UTC)
    if 'end_datetime' not in st.session_state:
        st.session_state.end_datetime = datetime.combine(default_end_datetime, time(0, 0)).replace(tzinfo=pytz.UTC)
    if 'log_scale' not in st.session_state:
        st.session_state.log_scale = True
    if 'bots_data' not in st.session_state:
        st.session_state.bots_data = {}

    # Initialize a specific dictionary for this page's selections if it doesn't exist
    if 'time_filter_selections' not in st.session_state:
        st.session_state.time_filter_selections = {
            'minutes': [],
            'hours': [],
            'days_of_week': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            'months': [],
            'years': [],
            'form_submitted_once': False  # Flag to check if form was submitted
        }

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
