"""
Module for handling column selection functionality in the Manual Annotation Tool.
"""

import streamlit as st
import pandas as pd
from typing import List, Tuple, Optional


def select_columns(
    df: pd.DataFrame, new_col_name: str, annotator_name: str
) -> Tuple[List[str], Optional[str], bool]:
    """
    Step 6: Select Columns to Display
    Allows the user to choose which columns to see while annotating.
    """
    st.header("Step 6: Select Columns to Display")

    flag_col = f"Invalid_{annotator_name}"
    possible_columns = [c for c in df.columns if c not in (new_col_name, flag_col)]

    # Use all possible columns as default if no previous selection exists.
    default_value = st.session_state.get("selected_columns", possible_columns)

    # Create the multiselect without manually reassigning the session state afterwards.
    selected_columns = st.multiselect(
        "Choose which columns to see while annotating:",
        options=possible_columns,
        default=default_value,
        # Optionally remove the key parameter so Streamlit auto-manages it
    )

    if not selected_columns:
        st.info("No columns selected. Please pick at least one.")
        st.stop()

    # Add sorting options
    st.subheader("Sorting Options")

    st.markdown(
        """
        You can choose to sort the data by a specific column. This will only affect the display order of the data.
        """
    )

    # Option to enable/disable sorting
    enable_sorting = st.checkbox("Enable sorting", value=False)

    # Column to sort by
    sort_column = None
    if enable_sorting and selected_columns:
        sort_column = st.selectbox(
            "Select column to sort by:", options=selected_columns
        )

    # Store in session state for persistence
    st.session_state.sort_column = sort_column
    st.session_state.enable_sorting = enable_sorting

    return selected_columns, sort_column, enable_sorting
