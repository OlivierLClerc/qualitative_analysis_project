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

    Args:
        df: The DataFrame to annotate
        new_col_name: The column name for annotations
        annotator_name: The annotator name

    Returns:
        A list of selected column names
    """
    st.header("Step 6: Select Columns to Display")

    flag_col = f"Unvalid_{annotator_name}"
    possible_columns = [c for c in df.columns if c not in (new_col_name, flag_col)]

    selected_columns = st.multiselect(
        "Choose which columns to see while annotating:",
        options=possible_columns,
        default=st.session_state.get("selected_columns", []),
        key="selected_columns",
    )

    # Don't update session state here - Streamlit does this automatically
    # when using the key parameter

    if not selected_columns:
        st.info("No columns selected. Please pick at least one.")
        st.stop()

    return selected_columns
