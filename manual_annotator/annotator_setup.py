"""
Module for handling annotator setup functionality in the Manual Annotation Tool.
"""

import streamlit as st
import pandas as pd
from typing import Tuple


def setup_annotator(
    df: pd.DataFrame,
    annotator_name: str,
    new_col_name: str,
    annotated_count: int,
    unannotated_count: int,
    total_count: int,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Step 3: Set Annotator Name
    Allows the user to set their name/initials and creates columns for annotations.

    Args:
        df: The DataFrame to annotate
        annotator_name: Current annotator name
        new_col_name: Current column name for annotations
        annotated_count: Number of annotated rows
        unannotated_count: Number of unannotated rows
        total_count: Total number of rows

    Returns:
        A tuple containing:
        - df: The updated DataFrame
        - annotator_name: The annotator name
        - new_col_name: The column name for annotations
    """
    st.header("Step 3: Set Annotator Name")

    annotator = st.text_input(
        "Annotator Name / Initials:",
        value=annotator_name,
        key="annotator_input",
    )

    st.markdown(
        "A new column will be created for your annotations if it doesn't exist (the column will be created as Rater_YourName)."
    )

    if annotator:
        annotator_name = annotator
        st.session_state.annotator_name = annotator_name

    # Create or re-use the column if user clicks confirm
    if new_col_name == "":
        if st.button("Confirm Annotator Name"):
            if annotator_name:
                candidate_col = f"Rater_{annotator_name}"
                flag_col = f"Unvalid_{annotator_name}"
                if candidate_col not in df.columns:
                    df[candidate_col] = pd.NA
                if flag_col not in df.columns:
                    df[flag_col] = False
                new_col_name = candidate_col

                # Preserve the counts in session state
                st.session_state.annotated_count = annotated_count
                st.session_state.unannotated_count = unannotated_count
                st.session_state.total_count = total_count

                st.session_state.new_col_name = new_col_name
                st.success(f"Created/Found columns '{candidate_col}' and '{flag_col}'!")
                st.rerun()

    if not new_col_name:
        st.stop()

    return df, annotator_name, new_col_name
