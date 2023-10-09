# External Library Imports
import os

import streamlit as st

from dotenv import load_dotenv
from sqlalchemy import create_engine

# Local Module Imports
from constants import FREQUENCY_MAPPING
from render import prices_page_desktop, prices_page_mobile, bots_page

# ------------ Session State Initialization ------------
def initialize_session_state():
    if 'dataframes_dict' not in st.session_state:
        st.session_state.dataframes_dict = {}
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = None
    if 'selected_freq' not in st.session_state or st.session_state.selected_freq not in FREQUENCY_MAPPING.values():
        st.session_state.selected_freq = "1H"  # Default to 1 hour
    if 'version' not in st.session_state:
        st.session_state.version = ''
    if 'engine' not in st.session_state:
            load_dotenv()
            database_password = os.environ.get('password')
            server_address = os.environ.get('serverip')
            connection_string = f"postgresql://postgres:{database_password}@{server_address}:5432/postgres"
            st.session_state.engine = create_engine(connection_string, echo=False)

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

    # Rest of your code
    pages = {
        "📈 Market Prices": prices_page_desktop if st.session_state.version == 'Desktop' else prices_page_mobile,
        "🤖 Bots": bots_page,
    }

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    pages[page](st.session_state.engine)

if __name__ == "__main__":
    main()