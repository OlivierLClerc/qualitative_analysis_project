"""
Module for handling dataset upload functionality in the Manual Annotation Tool.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Tuple

from qualitative_analysis import load_data


def upload_dataset(
    current_df: Optional[pd.DataFrame], original_df: Optional[pd.DataFrame]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Step 1: Upload CSV Dataset
    Uploads a dataset (CSV) via Streamlit's file uploader.

    Args:
        current_df: The current DataFrame or None
        original_df: The original DataFrame or None

    Returns:
        A tuple containing (current_df, original_df)
    """
    st.header("Step 1: Upload Your Data (CSV)")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="data_file")

    if uploaded_file is not None and current_df is None:
        # Load data via your custom function from qualitative_analysis
        loaded_df = load_data(
            uploaded_file, file_type="csv", delimiter=";"
        ).reset_index(drop=True)

        # Store in session state
        current_df = loaded_df.copy()
        original_df = loaded_df.copy()

        st.session_state.df = current_df
        st.session_state.original_df = original_df

        st.success(f"Data loaded successfully ({len(current_df)} rows).")

    if current_df is None:
        st.stop()
        return None, None

    # Show a preview of the data
    st.write("Here are the first 5 rows of your data:")
    st.dataframe(current_df.head())

    return current_df, original_df
