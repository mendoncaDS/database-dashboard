# External Library Imports
import pytz

import time as t
import pandas as pd
import streamlit as st
import plotly.express as px

from datetime import datetime, timedelta, time

# Local Module Imports
from constants import FREQUENCY_MAPPING
from auxiliary import get_human_readable_freq, fetch_and_plot_data, unique_symbol_freqs, plot_in_placeholder, process_indicators

# --------------- Page Rendering Functions ---------------
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

    with st.sidebar.form(key='fetch_from_database_form'):
        st.session_state.price_plotted = False
        # Symbol and Frequency selectors inside the form
        selected_symbol = symbol_selector(engine)
        selected_freq_display = frequency_selector()

        # Set default value for date selector to 6 months ago
        #six_months_ago = datetime.today() - pd.DateOffset(months=6)
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Select Start Date", st.session_state.start_datetime)
        with col2:
            # End Time Selector
            end_date = st.date_input("Select End Date", st.session_state.end_datetime)

        # Convert the selected_date and end_date to datetime objects and localize to UTC
        start_datetime = datetime.combine(start_date, time(0, 0)).replace(tzinfo=pytz.UTC)
        end_datetime = datetime.combine(end_date, time(0, 0)).replace(tzinfo=pytz.UTC)

        # The submit button for the form
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button(label='Load Data')
        with col2:
            log_scale = st.checkbox("Log Y", value=st.session_state.log_scale)

    # Fetch and plot the data using the selected start and end timestamps
    if submit_button or not st.session_state.price_plotted:
        st.session_state.price_plotted = True
        st.session_state.selected_symbol = selected_symbol
        st.session_state.selected_freq = FREQUENCY_MAPPING[selected_freq_display]
        st.session_state.start_datetime = start_datetime
        st.session_state.end_datetime = end_datetime
        st.session_state.log_scale = log_scale

        start_time = t.time()  # Start timing here
        fetch_and_plot_data(graph_name_placeholder, graph_placeholder, start_datetime, end_datetime, start_time, engine)
    if submit_button:
        st.experimental_rerun()
    
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

        st.session_state.price_plotted = False
        # Symbol and Frequency selectors inside the form
        selected_symbol = symbol_selector(engine)
        selected_freq_display = frequency_selector()

        # Set default value for date selector to 6 months ago

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Select Start Date", st.session_state.start_datetime)
        with col2:
            # End Time Selector
            end_date = st.date_input("Select End Date", st.session_state.end_datetime)

        # Convert the selected_date and end_date to datetime objects and localize to UTC
        start_datetime = datetime.combine(start_date, time(0, 0)).replace(tzinfo=pytz.UTC)
        end_datetime = datetime.combine(end_date, time(0, 0)).replace(tzinfo=pytz.UTC)

        # Adjusting column widths to force side-by-side layout
        col1, col2 = st.columns([1,1])
        with col1:
            submit_button = st.form_submit_button(label='Load Data')
        with col2:
            log_scale = st.checkbox("Log Y", value=st.session_state.log_scale)

    # Check if the form was submitted
    if submit_button or not st.session_state.price_plotted:
        st.session_state.price_plotted = True
        st.session_state.selected_symbol = selected_symbol
        st.session_state.selected_freq = FREQUENCY_MAPPING[selected_freq_display]
        st.session_state.start_datetime = start_datetime
        st.session_state.end_datetime = end_datetime
        st.session_state.log_scale = log_scale
        start_time = t.time()  # Start timing here
        fetch_and_plot_data(graph_name_placeholder, graph_placeholder, start_datetime, end_datetime, start_time, engine)
    if submit_button:
        st.experimental_rerun()

    # Update the graph_name_placeholder with the current symbol and frequency after the form is processed
    if st.session_state.selected_symbol is not None:
        human_readable_freq = get_human_readable_freq(st.session_state.selected_freq)
        graph_name_placeholder.write(f"💲 {st.session_state.selected_symbol} - {human_readable_freq}")
    else:
        graph_name_placeholder.write("💲")

def indicators_page(engine):
    st.title("📊 Indicators")

    # Placeholder for the graph
    graph_placeholder = st.empty()

    # Dropdown for indicator selection
    available_indicators = ["SMA", "EMA", "WMA", "HMA", "DEMA", "TEMA", "TRIMA", "KAMA", "ZLMA", "ALMA", "BBANDS"]

    # Initialize the indicators list if it doesn't exist
    if not hasattr(st.session_state, "indicators_list"):
        st.session_state.indicators_list = []

    # Create legend identifiers for the indicators
    legend_identifiers = []
    if st.session_state.indicators_list:
        counter = {indicator['indicator']: 0 for indicator in st.session_state.indicators_list}
        for item in st.session_state.indicators_list:
            counter[item['indicator']] += 1
            legend_identifiers.append(f"{item['indicator']} {counter[item['indicator']]}")

    # Create columns for forms and indicator table
    col1, col2 = st.columns(2)

    selected_indicator = col1.selectbox("Add an Indicator", available_indicators)

    # Display appropriate parameters input UI based on the indicator selected
    with col1.form(key='indicator_form'):
        params = {}
        if selected_indicator in ["SMA", "EMA", "WMA", "HMA", "DEMA", "TEMA", "TRIMA", "KAMA", "ZLMA", "ALMA"]:
            params["period"] = st.number_input(f"Enter {selected_indicator} Period", value=20, min_value=1)
        elif selected_indicator == "BBANDS":
            inner_col1, inner_col2 = st.columns(2)
            with inner_col1:
                params["length"] = st.number_input("Length", value=20, min_value=1)
            with inner_col2:
                params["std"] = st.number_input("Number of standard deviations", value=2.0, min_value=0.5, step=0.1)
        add_indicator = st.form_submit_button(label=f"Add {selected_indicator}")

    # Add the selected indicator and its parameters to the indicators list
    if add_indicator:
        st.session_state.indicators_list.append({"indicator": selected_indicator, "params": params})
        # Update the legend_identifiers list after adding a new indicator
        counter = {indicator['indicator']: 0 for indicator in st.session_state.indicators_list}
        legend_identifiers = []
        for item in st.session_state.indicators_list:
            counter[item['indicator']] += 1
            legend_identifiers.append(f"{item['indicator']} {counter[item['indicator']]}")

    # Form for deleting selected indicators in the left column
    with col1.form(key='delete_form'):
        # Allow user to select entries to delete
        if st.session_state.indicators_list:
            indices_to_delete = st.multiselect("Delete an Indicator:", options=list(range(len(st.session_state.indicators_list))), format_func=lambda x: legend_identifiers[x])
            delete_button = st.form_submit_button("Delete Selected Indicators")
            
            if delete_button:
                # Delete the selected indices from the list
                for index in sorted(indices_to_delete, reverse=True):
                    del st.session_state.indicators_list[index]
                # Clear the selection after deletion
                indices_to_delete.clear()
                st.experimental_rerun()  # To ensure the UI updates

    # Display the selected indicators in a table in the right column
    if st.session_state.indicators_list:
        with col2:
            st.write("Current Indicators")
            
            df = pd.DataFrame(st.session_state.indicators_list)
            df['Indicator'] = legend_identifiers

            # Extract the parameters and their values
            indicator_names = df['Indicator'].tolist()
            list_of_dicts = df['params'].tolist()
            
            # Construct rows for multi-index
            rows = []
            for indicator, params in zip(indicator_names, list_of_dicts):
                for param, value in params.items():
                    rows.append((indicator, param, value))
            
            # Create the multi-index DataFrame
            multi_df = pd.DataFrame(rows, columns=["Indicator", "Parameter", "Value"]).set_index(["Indicator", "Parameter"])
            
            # Group by and unstack to get the desired multi-index structure
            multi_df = multi_df.groupby(level=[0, 1]).first().unstack().stack()
            
            st.write(multi_df.to_html(), unsafe_allow_html=True)

    # If there's data available, process and plot it
    if hasattr(st.session_state, "graph_data") and st.session_state.graph_data is not None:
        df = process_indicators(st.session_state.graph_data, st.session_state.indicators_list)
        plot_dataframe(df, graph_placeholder)

def bots_page(engine):
    st.title("🤖 Bots")
    st.subheader("Coming soon!")

def get_pages(session_state):
    
    return {
        "📈 Market Prices": (prices_page_desktop if session_state.version == 'Desktop' else prices_page_mobile),
        "📊 Indicators": indicators_page,
        "🤖 Bots": bots_page,
    }

# -------------- Widget Rendering Functions --------------

def symbol_selector(engine):
    """
    Function to display only the symbol dropdown.
    """
    default_symbol_index = st.session_state.unique_symbols.index(st.session_state.selected_symbol)

    selected_symbol = st.selectbox(
        "Select a Symbol",
        st.session_state.unique_symbols,
        index=default_symbol_index,
    )

    return selected_symbol

def frequency_selector():
    """
    Function to display the frequency dropdown outside the form.
    """

    REVERSE_FREQUENCY_MAPPING = {value: key for key, value in FREQUENCY_MAPPING.items()}

    selected_freq_display_format = REVERSE_FREQUENCY_MAPPING.get(st.session_state.selected_freq, "1 hour")  # default to "1 hour" if not found

    all_freq_display_formats = list(FREQUENCY_MAPPING.keys())

    current_index = all_freq_display_formats.index(selected_freq_display_format)

    selected_freq_display = st.selectbox("Select Frequency", all_freq_display_formats, index=current_index)
    
    return selected_freq_display


def plot_dataframe(df, placeholder):
    """ Plot the dataframe's columns using plotly. """
    fig = px.line(df)

    # Check if logarithmic scale is selected
    if st.session_state.get('log_scale', False):
        fig.update_yaxes(type="log")

    # Update the layout to adjust the height and width
    fig.update_layout(autosize=True, height=650)  # Adjust as needed

    placeholder.plotly_chart(fig, use_container_width=True, theme="streamlit")

