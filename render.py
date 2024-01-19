# External Library Imports
import pytz

import time as t
import numpy as np
import pandas as pd
import vectorbt as vbt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, time, timedelta
from plotly.subplots import make_subplots

# Local Module Imports
from constants import FREQUENCY_MAPPING
from auxiliary import get_human_readable_freq, fetch_and_plot_data, plot_in_placeholder, process_indicators, resample_data, unique_bots, update_bot_data, load_symbol_data, get_last_timestamp



# --------------- Pre Aux Temp Functions ---------------

def fetch_missing_price_data(engine, symbol, start_datetime):
    
    # before populating variable local_data, check if symbol is in dict's keys
    if symbol not in st.session_state['dataframes_dict']:
        st.session_state['dataframes_dict'][symbol] = load_symbol_data(symbol, start_datetime, datetime.now(pytz.timezone('UTC')), engine).iloc[:-1]
        return

    local_data = st.session_state['dataframes_dict'][symbol]
    local_min_ts = local_data.index.min()
    local_max_ts = local_data.index.max()

    last_available_timestamp = get_last_timestamp(symbol, engine).replace(tzinfo=pytz.UTC)

    # Fetch older data if needed
    if start_datetime < local_min_ts:
        older_data = load_symbol_data(symbol, start_datetime, local_min_ts, engine)
        older_data = older_data.iloc[:-1]
        st.session_state['dataframes_dict'][symbol] = pd.concat([older_data, local_data])

    # Fetch newer data if needed
    if last_available_timestamp > local_max_ts:
        newer_data = load_symbol_data(symbol, local_max_ts, last_available_timestamp, engine)
        newer_data = newer_data.iloc[1:-1]
        st.session_state['dataframes_dict'][symbol] = pd.concat([local_data, newer_data])
    
    print("Fetch missing price DONE")


# --------------- Page Rendering Functions ---------------

def prices_page(engine):
    st.title("📈 Market Visualization")

    # Placeholder for the graph
    graph_placeholder = st.empty()

    # Process and plot data if available
    if hasattr(st.session_state, "graph_data") and st.session_state.graph_data is not None:
        df = process_indicators(st.session_state.graph_data, st.session_state.indicators_list)
        plot_trends_and_oscillators(df, graph_placeholder)  # Call the plot function

    # Initialize the indicators list if it doesn't exist
    if not hasattr(st.session_state, "indicators_list"):
        st.session_state.indicators_list = []

    # Data fetching form
    with st.form(key='fetch_from_database_form'):
        st.session_state.price_plotted = False
        # Symbol and Frequency selectors inside the form
        selected_symbol = symbol_selector(engine)
        selected_freq_display = frequency_selector()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Select Start Date", st.session_state.start_datetime)
        with col2:
            end_date = st.date_input("Select End Date", st.session_state.end_datetime)

        # Convert to datetime objects and localize to UTC
        start_datetime = datetime.combine(start_date, time(0, 0)).replace(tzinfo=pytz.UTC)
        end_datetime = datetime.combine(end_date, time(0, 0)).replace(tzinfo=pytz.UTC)

        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button(label='Load Data')
        with col2:
            log_scale = st.checkbox("Log Y", value=st.session_state.log_scale)

    # Fetch and plot the data using the selected timestamps
    if submit_button or not st.session_state.price_plotted:
        st.session_state.price_plotted = True
        st.session_state.selected_symbol = selected_symbol
        st.session_state.selected_freq = FREQUENCY_MAPPING[selected_freq_display]
        st.session_state.start_datetime = start_datetime
        st.session_state.end_datetime = end_datetime
        st.session_state.log_scale = log_scale

        fetch_and_plot_data(start_datetime, end_datetime, engine)
        if submit_button:
            st.rerun()

    # Dropdown for indicator selection
    available_indicators = ["SMA", "EMA", "WMA", "HMA", "DEMA", "TEMA", "TRIMA", "KAMA", "ZLMA", "ALMA", "BBANDS", "RSI", "STOCH", "MACD"]

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

        # For indicators that only require a period
        if selected_indicator in ["SMA", "EMA", "WMA", "HMA", "DEMA", "TEMA", "TRIMA", "KAMA", "ZLMA", "ALMA", "RSI"]:
            params["period"] = st.number_input(f"Enter {selected_indicator} Period", value=20, min_value=1)

        elif selected_indicator == "BBANDS":
            # For Bollinger Bands, which require length and std deviation
            inner_col1, inner_col2 = st.columns(2)
            with inner_col1:
                params["length"] = st.number_input("Length", value=20, min_value=1)
            with inner_col2:
                params["std"] = st.number_input("Std Dev", value=2.0, min_value=0.1, step=0.1)

        elif selected_indicator == "STOCH":
            # For Stochastic oscillator, which requires several parameters
            params["fast_k_period"] = st.number_input("Fast K Period", value=14, min_value=1)
            params["slow_k_period"] = st.number_input("Slow K Period", value=3, min_value=1)
            params["slow_d_period"] = st.number_input("Slow D Period", value=3, min_value=1)

        elif selected_indicator == "MACD":
            # For MACD, which requires fast period, slow period, and signal period
            inner_col1, inner_col2, inner_col3 = st.columns(3)
            with inner_col1:
                params["fast_period"] = st.number_input("Fast Period", value=12, min_value=1)
            with inner_col2:
                params["slow_period"] = st.number_input("Slow Period", value=26, min_value=1)
            with inner_col3:
                params["signal_period"] = st.number_input("Signal Period", value=9, min_value=1)

        add_indicator = st.form_submit_button(label=f"Add {selected_indicator}")

    # Add the selected indicator and its parameters to the indicators list
    if add_indicator:
        # Construct the new indicator entry
        new_indicator_entry = {"indicator": selected_indicator, "params": params}

        # Check if it's already in the list
        if new_indicator_entry in st.session_state.indicators_list:
            # Inform the user that this exact indicator configuration already exists
            st.error(f"The indicator {selected_indicator} with the same parameters already exists. Please choose different parameters or select another indicator.")
        else:
            # If it's a unique entry, append it to the list and update the legends
            st.session_state.indicators_list.append(new_indicator_entry)

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
                st.rerun()  # To ensure the UI updates

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
        plot_trends_and_oscillators(df, graph_placeholder)  # here we call the new plot function

def plot_trends_and_oscillators(df, placeholder):
    print("plot_trends_and_oscillators")

    # Separate the trend and oscillator indicators based on the column naming
    trend_cols = [col for col in df.columns if col.startswith('trend_') or col == 'close']
    osc_cols = [col for col in df.columns if col.startswith('osc_')]

    trend_data = df[trend_cols]
    osc_data = df[osc_cols]

    # Define the heights
    main_plot_ratio = 4  # The main plot is 2.5 times the height of each oscillator plot
    per_oscillator_height = 150  # Height for each oscillator subplot
    main_plot_height = int(main_plot_ratio * per_oscillator_height)  # Based on the ratio

    # First, let's create a function that extracts the base name and group identifier.
    def extract_base_name(column):
        parts = column.split('_')
        # The base name is the part after 'osc', and the group identifier is the last part.
        base_name = parts[1]  # This is the actual indicator name (e.g., 'MACD' in 'osc_MACD_signal_1')
        group_identifier = parts[-1]  # This is the group identifier (e.g., '1' in 'osc_MACD_signal_1')
        return f"{base_name}_{group_identifier}"  # Combine them for grouping

    # Apply the function to all oscillator columns to get the correct base names.
    osc_base_names = set(extract_base_name(col) for col in osc_cols)

    # Create a dictionary where keys are base names and values are lists of related columns.
    osc_groups = {base_name: [] for base_name in osc_base_names}
    for col in osc_cols:
        base_name = extract_base_name(col)
        osc_groups[base_name].append(col)

    num_oscillator_groups = len(osc_groups)

    indicator_line_style = {"width": 1.5}  # thinner lines for indicators
    close_line_style = {"width": 2}  # default style, more visible
    indicator_opacity = 0.75  # transparency level for indicators

    # Adjust the total figure height based on the number of oscillator groups.
    total_fig_height = main_plot_height + num_oscillator_groups * per_oscillator_height

    # Create subplots based on the number of oscillator groups
    if osc_groups:
        # Define the relative heights of the subplots
        subplot_heights = [main_plot_ratio] + [1] * num_oscillator_groups

        # Create subplots with custom heights
        fig = make_subplots(
            rows=1 + num_oscillator_groups, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.04,
            subplot_titles=(["Trend Indicators"] + [name.replace('osc_', '') for name in osc_groups.keys()]),
            row_heights=subplot_heights
        )

        for col in trend_data.columns:
            # Remove 'trend_' prefix for legend
            legend_name = col.replace('trend_', '') if col.startswith('trend_') else col
            line_style = close_line_style if col == 'close' else indicator_line_style
            trace_opacity = 1 if col == 'close' else indicator_opacity  # full opacity for 'close', custom for others

            fig.add_trace(go.Scatter(
                x=df.index, 
                y=trend_data[col], 
                mode='lines', 
                name=legend_name, 
                line=line_style,  # Apply the customized style here
                opacity=trace_opacity  # setting the opacity at the trace level
            ), row=1, col=1)

        # Now, we loop through the oscillator groups, not individual columns
        osc_row_idx = 2  # initial row index for oscillator plots
        for osc_name, columns in osc_groups.items():
            for col in columns:
                # The legend name should be the full column name without 'osc_' prefix.
                legend_name = col.replace('osc_', '')  # Full indicator name excluding 'osc_'
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=osc_data[col], 
                    mode='lines', 
                    name=legend_name, 
                    line=indicator_line_style,  # indicators style
                    opacity=indicator_opacity  # setting the opacity at the trace level
                ), row=osc_row_idx, col=1)
            osc_row_idx += 1  # move to the next subplot for the next group



        fig.update_xaxes(title_text="Date", row=1 + num_oscillator_groups, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)

        for idx in range(2, 2 + num_oscillator_groups):
            fig.update_yaxes(title_text="Value", row=idx, col=1)

        if st.session_state.get('log_scale', True):
            fig.update_yaxes(type="log", row=1, col=1)
        
        print("A")

    else:
        # If there are no oscillators, create a single plot for the main plot area
        print("B")
        fig = go.Figure()

        # Add trend data to the plot
        for col in trend_data.columns:
            # Remove 'trend_' prefix for legend
            legend_name = col.replace('trend_', '') if col.startswith('trend_') else col
            line_style = close_line_style if col == 'close' else indicator_line_style
            trace_opacity = 1 if col == 'close' else indicator_opacity  # full opacity for 'close', custom for others

            fig.add_trace(go.Scatter(
                x=df.index, 
                y=trend_data[col], 
                mode='lines', 
                name=legend_name, 
                line=line_style,  # Apply the customized style here
                opacity=trace_opacity  # setting the opacity at the trace level
            ))  # No row/col references here
        
        if st.session_state.get('log_scale', True):
            fig.update_yaxes(type="log")


    # Update layout
    fig.update_layout(
        height=total_fig_height, 
        width=1000, 
        # make this if statement within the string definition
        # if st.session_state.selected_symbol is not None:
            # human_readable_freq = get_human_readable_freq(st.session_state.selected_freq)
            # graph_name_placeholder.write(f"💲 {st.session_state.selected_symbol} - {human_readable_freq}")
        #
        title_text=(f"💲 {st.session_state.selected_symbol} - {get_human_readable_freq(st.session_state.selected_freq)}") if (st.session_state.selected_symbol is not None) else None
    )

    # Display the figure
    placeholder.plotly_chart(fig, use_container_width=True)


def bots_page(engine):
    st.title("🤖 Bots")

    # Create tabs for in-page navigation
    tab1, tab2 = st.tabs(["Visualize Backtest", "Compare Bots"])

    with tab1:
        if 'portfolio' not in st.session_state:
            st.session_state.pf = None  # Initialize if not present

        # Initialize first load flag in session state
        if 'bots_page_first_load' not in st.session_state:
            selected_bot = st.session_state.unique_bots_list[0]
            st.session_state.bots_page_first_load = True
            # Set default start date on first load
            st.session_state.selected_begin_date = datetime.strptime(st.session_state.bots_data_dict[selected_bot]["bt_end_date"], "%Y-%m-%d")

        with st.form(key='bot_selection_form'):
            selected_bot = st.selectbox("Select a bot:", st.session_state.unique_bots_list, index=0)

            col1, col2 = st.columns(2)
            with col1:
                # Use session state to retain start date value after first load
                st.session_state.selected_begin_date = st.date_input("Select backtest start date:", value=st.session_state.selected_begin_date)
            with col2:
                # For end date, always use the current or selected date
                st.session_state.selected_end_date = st.date_input("Select backtest end date:", value=datetime.today().date()+timedelta(days=1))
            submit_button = st.form_submit_button(label='Update Backtest')

        # If there is no portfolio in session state or submit button is pressed
        if st.session_state.pf is None or submit_button:
            st.session_state.bots_page_first_load = False

            update_bot_data(engine, selected_bot)

            current_bot_data = st.session_state.bots_data_dict[selected_bot]["data"]
            current_bot_start = current_bot_data["timestamp"].min().replace(tzinfo=pytz.UTC)
            current_bot_end = current_bot_data["timestamp"].max()
            current_bot_symbol = st.session_state.bots_data_dict[selected_bot]["symbol"]

            fetch_missing_price_data(engine, current_bot_symbol, current_bot_start)

            processed_bot_data = current_bot_data[["timestamp", "position"]]
            processed_bot_data.set_index("timestamp", inplace=True)

            bot_openprice_data = st.session_state.dataframes_dict[current_bot_symbol][current_bot_start:]["open"]
            bot_openprice_data.index = bot_openprice_data.index.tz_localize(None)
            bot_openprice_data = bot_openprice_data.resample(f"1H").first()
            processed_bot_data = pd.concat([processed_bot_data, bot_openprice_data], axis=1)
            processed_bot_data["position"].ffill(inplace=True)
            processed_bot_data.dropna(inplace=True)
            processed_bot_data["entries"] = processed_bot_data["position"] == 1

            processed_bot_data = processed_bot_data[st.session_state.selected_begin_date:min(current_bot_end.to_pydatetime(), pd.to_datetime(st.session_state.selected_end_date))]

            # Portfolio creation
            st.session_state.pf = vbt.Portfolio.from_signals(
                processed_bot_data["open"],
                processed_bot_data["entries"],
                ~processed_bot_data["entries"],
                init_cash=100,
                freq="1H",
            )

        # Always try to render the visualization from session state portfolio
        if st.session_state.pf:
            pf_stats = st.session_state.pf.stats()
            pf_stats_str = pf_stats.astype(str)

            st.subheader("Backtest Information")
            with st.expander("Show Backtest Info"):
                col1, col2 = st.columns(2)
                with col1:
                    pf_stats_str.name = selected_bot
                    pf_stats_df = pf_stats_str.to_frame(name='Value')
                    st.table(pf_stats_df)
                with col2:
                    st.write(f"Bot Name: {selected_bot}")
                    st.write(f"Traded Market: {st.session_state.bots_data_dict[selected_bot]['symbol']} - Binance")
                    st.write(f"Training Beginning: {st.session_state.bots_data_dict[selected_bot]['bt_begin_date']}")
                    st.write(f"Training End: {st.session_state.bots_data_dict[selected_bot]['bt_end_date']}")

            st.subheader("Backtest visualization")
            with st.expander("Show backtest visualization", expanded=True):
                st.plotly_chart(st.session_state.pf.plot(subplots=[
                    "trades",
                    "trade_pnl",
                    "cum_returns",
                    "underwater",
                    "net_exposure",
                ]), use_container_width=True)

            st.markdown("---")

    with tab2:
        if 'pf_list' not in st.session_state:
            st.session_state.pf_list = []  # Initialize if not present
            
        st.subheader("Bot Comparison")

        with st.form(key='bot_comparison_form'):
            # Multi-selector for bots
            selected_bots = st.multiselect("Select Bots:", st.session_state.unique_bots_list, default=[st.session_state.unique_bots_list[0], st.session_state.unique_bots_list[1]])
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Select start date for comparison:", value=datetime.today().date() - timedelta(days=180))
            with col2:
                end_date = st.date_input("Select end date for comparison:", value=datetime.today().date())
            submit_button = st.form_submit_button(label='Run Comparison')


        if submit_button:
            if len(selected_bots) < 2:
                st.warning("Please select at least two bots for comparison.")
            else:
                bot_data_list = []
                st.session_state.pf_list = []  # Reset pf_list in session state
                for bot in selected_bots:
                    bot_data = process_and_backtest_bot(engine, bot, start_date, end_date)
                    bot_start = bot_data.index.min()
                    bot_data_list.append((bot, bot_data, bot_start))

                # Find the latest start date among all bots
                latest_start_date = max(bot_start for _, _, bot_start in bot_data_list)

                # Adjust data and create portfolios
                for bot, bot_data, _ in bot_data_list:
                    adjusted_bot_data = bot_data[latest_start_date:]
                    pf = vbt.Portfolio.from_signals(
                        adjusted_bot_data["open"],
                        adjusted_bot_data["entries"],
                        ~adjusted_bot_data["entries"],
                        init_cash=100,
                        freq="1H",
                    )
                    st.session_state.pf_list.append((bot, pf))

        # Use pf_list from session state for subsequent operations
        if st.session_state.pf_list:
            # Merge the stats of all bots from pf_list in session state
            merged_df = pd.concat([pf.stats() for _, pf in st.session_state.pf_list], axis=1)
            merged_df.columns = [bot for bot, _ in st.session_state.pf_list]
            
            # Display backtest results
            st.table(merged_df)


def process_and_backtest_bot(engine, bot_name, start_date, end_date):
    update_bot_data(engine, bot_name)

    current_bot_data = st.session_state.bots_data_dict[bot_name]["data"]
    current_bot_start = current_bot_data["timestamp"].min().replace(tzinfo=pytz.UTC)
    current_bot_end = current_bot_data["timestamp"].max()
    current_bot_symbol = st.session_state.bots_data_dict[bot_name]["symbol"]

    fetch_missing_price_data(engine,current_bot_symbol,current_bot_start)

    processed_bot_data = current_bot_data[["timestamp","position"]]

    processed_bot_data.set_index("timestamp", inplace=True)


    bot_openprice_data = st.session_state.dataframes_dict[current_bot_symbol][current_bot_start:]["open"]
    bot_openprice_data.index = bot_openprice_data.index.tz_localize(None)
    bot_openprice_data = bot_openprice_data.resample(f"1H").first()
    processed_bot_data = pd.concat([processed_bot_data, bot_openprice_data], axis=1)
    processed_bot_data["position"].ffill(inplace=True)
    processed_bot_data.dropna(inplace=True)
    processed_bot_data["entries"] = processed_bot_data["position"] == 1

    processed_bot_data = processed_bot_data[start_date:min(current_bot_end.to_pydatetime(),pd.to_datetime(end_date))]

    return processed_bot_data


    # Portfolio creation code remains unchanged


    return pf


def time_filtered_returns(engine):

    def determine_resampling_freq(mins, hrs, days, months):
        """
        Determine the resampling frequency based on the selected timeframes.
        """
        if len(mins) > 0:
            return "1T"
        elif len(hrs) > 0:
            return "1H"
        elif len(days) > 0:
            return "1D"
        elif len(months) > 0:
            return "1M"
        else:
            return "1Y"

    def analyze_time_filtered_returns(ohlcv_df, mins, hrs, days, months, yrs):
        # Filter data based on user's selection
        ohlcv_df = ohlcv_df[
            (ohlcv_df.index.minute.isin(mins) if len(mins) > 0 else True) &
            (ohlcv_df.index.hour.isin(hrs) if len(hrs) > 0 else True) &
            (ohlcv_df.index.day_name().isin(days) if len(days) > 0 else True) &
            (ohlcv_df.index.month.isin(months) if len(months) > 0 else True) &
            (ohlcv_df.index.year.isin(yrs) if len(yrs) > 0 else True)
        ]
        
        # Calculate the daily returns
        ohlcv_df['returns'] = ((ohlcv_df['close'] - ohlcv_df['open']) / ohlcv_df['open'])*100
        
        # Create a list of columns based on the user's selections. This list will determine the columns used for grouping.
        ohlcv_df['minute'] = ohlcv_df.index.minute
        ohlcv_df['hour'] = ohlcv_df.index.hour
        ohlcv_df['day_of_week'] = ohlcv_df.index.day_name()
        ohlcv_df['month'] = ohlcv_df.index.month
        ohlcv_df['year'] = ohlcv_df.index.year

        # Now we're sure all these columns are in the dataframe, regardless of the filtering options used.
        group_by_columns = ['minute', 'hour', 'day_of_week', 'month', 'year']
        
        # Define a new list based on the actual selections made by the user. 
        # This list will respect the user's choice of granularity.
        selected_columns = []
        if mins:
            selected_columns.append('minute')
        if hrs:
            selected_columns.append('hour')
        if days:
            selected_columns.append('day_of_week')
        if months:
            selected_columns.append('month')
        if yrs:
            selected_columns.append('year')

        # Determine the primary axis based on the most granular selection made by the user.
        if selected_columns:
            primary_axis = selected_columns[0]  # The first selection is the most granular
        else:
            primary_axis = 'returns'  # default if no specific time unit is selected

        if 'day_of_week' in ohlcv_df.columns:
            # Define the correct order of the days
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Convert 'day_of_week' to a category with ordered levels
            ohlcv_df['day_of_week'] = pd.Categorical(ohlcv_df['day_of_week'], categories=days_order, ordered=True)

        # No changes needed in the DataFrame creation
        result_df = ohlcv_df[['returns'] + group_by_columns]

        return result_df, primary_axis

    def plot_returns(returns, primary_axis, graph_placeholder):
        if not returns.empty:
            fig_violin = px.box(
                returns,
                x=primary_axis,
                y='returns',
                height=700,
            )
            graph_placeholder.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.warning("No data available after filtering. Please adjust your selection.")

    def analyze_returns_plot(ohlcv_df, graph_placeholder, minutes, hours, days_of_week, months, years):
        # Determine the resampling frequency
        freq = determine_resampling_freq(minutes, hours, days_of_week, months)

        # Resample the data
        ohlcv_df = resample_data(ohlcv_df, freq)

        returns, primary_axis = analyze_time_filtered_returns(ohlcv_df, minutes, hours, days_of_week, months, years)

        st.session_state.last_return_analysis_res = {}
        st.session_state.last_return_analysis_res["returns"] = returns
        st.session_state.last_return_analysis_res["primary_axis"] = primary_axis

        plot_returns(returns, primary_axis, graph_placeholder)

    # Fetching the 1m granular data for the selected symbol from session_state
    if st.session_state.selected_symbol in st.session_state.dataframes_dict:
        ohlcv_df = st.session_state.dataframes_dict[st.session_state.selected_symbol]
    else:
        st.warning("No data available.")
        return

    st.title('🕵️‍♂️ Time Filtered Returns Analysis')

    if st.session_state.selected_symbol is not None:
        st.write(f"💲 {st.session_state.selected_symbol}")

    # Placeholder for the graph
    graph_placeholder = st.empty()

    if hasattr(st.session_state, "last_return_analysis_res"):
        plot_returns(st.session_state.last_return_analysis_res["returns"], st.session_state.last_return_analysis_res["primary_axis"], graph_placeholder)

    st.write("Customize the Filter")
    with st.form(key='return_analysis_form'):
        # Use session state variables in the form
        minutes = st.multiselect("Minute", list(range(60)), default=st.session_state.time_filter_selections['minutes'])
        hours = st.multiselect("Hour", list(range(24)), default=st.session_state.time_filter_selections['hours'])
        days_of_week = st.multiselect("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], default=st.session_state.time_filter_selections['days_of_week'])
        months = st.multiselect("Month", list(range(1, 13)), default=st.session_state.time_filter_selections['months'])
        years = st.multiselect("Year", list(range(2017, datetime.now().year+1)), default=st.session_state.time_filter_selections['years'])

        submitted = st.form_submit_button("Analyze Returns")

    # Action upon form submission
    if submitted or not st.session_state.time_filter_selections['form_submitted_once']:
        st.session_state.time_filter_selections['minutes'] = minutes
        st.session_state.time_filter_selections['hours'] = hours
        st.session_state.time_filter_selections['days_of_week'] = days_of_week
        st.session_state.time_filter_selections['months'] = months
        st.session_state.time_filter_selections['years'] = years
        st.session_state.time_filter_selections['form_submitted_once'] = True

        analyze_returns_plot(ohlcv_df, graph_placeholder, minutes, hours, days_of_week, months, years)
    if submitted:
        st.rerun()


def get_pages(session_state):
    
    return {
        "🤖 Bots": bots_page,
        "📈 Market Prices": prices_page,
        "🕵️‍♂️ Filtered Returns Analysis": time_filtered_returns,
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

