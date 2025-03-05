"""
Module for handling column selection functionality in the Manual Annotation Tool.
"""

import streamlit as st
import pandas as pd
from typing import List


def select_columns(
    df: pd.DataFrame, new_col_name: str, annotator_name: str
) -> List[str]:
    """
    Step 6: Select Columns to Display
    Allows the user to choose which columns to see while annotating.
    """
    st.header("Step 6: Select Columns to Display")

    flag_col = f"Unvalid_{annotator_name}"
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

    return selected_columns
