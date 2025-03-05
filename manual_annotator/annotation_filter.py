"""
Module for handling annotation filtering functionality in the Manual Annotation Tool.
"""

import streamlit as st
import pandas as pd
from typing import Tuple, List


def filter_annotations(
    df: pd.DataFrame, annotation_columns: List[str]
) -> Tuple[pd.DataFrame, List[str], int, int, int]:
    """
    Step 2: Filter by Existing Annotation Columns
    Identifies rows that have non-null values in the selected annotation columns,
    but keeps all rows in the dataframe.

    Args:
        df: The DataFrame to filter
        annotation_columns: List of column names that contain annotations

    Returns:
        A tuple containing:
        - df: The original DataFrame (unchanged)
        - annotation_columns: Updated list of annotation columns
        - annotated_count: Number of annotated rows
        - unannotated_count: Number of unannotated rows
        - total_count: Total number of rows
    """
    st.header("Step 2: (Optional) Filter by Existing Annotation Columns")

    # Initialize the session state key to an empty list if it doesn't exist
    if "selected_annotation_cols" not in st.session_state:
        st.session_state["selected_annotation_cols"] = []

    # Create the multiselect with an empty default (if nothing has been selected yet)
    selected_annotation_cols: List[str] = st.multiselect(
        label="Select column(s) that contain existing human annotations:",
        options=df.columns,
        default=st.session_state["selected_annotation_cols"],
        help="Only rows with non-null values in ALL selected columns will be shown during annotation.",
        key="annotation_cols_key",
    )

    # Initialize counters
    total_count = len(df)
    annotated_count = 0
    unannotated_count = 0
    annotated_indices = []

    # Only calculate these if columns are chosen
    if selected_annotation_cols:
        # Identify rows that have non-null values in all of the selected columns
        annotated_mask = df[selected_annotation_cols].notna().all(axis=1)
        annotated_indices = df[annotated_mask].index.tolist()

        annotated_count = len(annotated_indices)
        unannotated_count = total_count - annotated_count

        # Store in session state (if needed later)
        st.session_state["annotated_indices"] = annotated_indices
        st.session_state["annotated_count"] = annotated_count
        st.session_state["unannotated_count"] = unannotated_count
        st.session_state["total_count"] = total_count

        # Show feedback
        if annotated_count == 0:
            # Show a warning instead of stopping, so user can keep adjusting columns
            st.warning(
                f"No rows have annotations in **all** of these columns: "
                f"{', '.join(selected_annotation_cols)}. "
                "Please adjust your selection."
            )
        else:
            st.success(
                f"Identified {annotated_count} rows with non-null values in "
                f"{', '.join(selected_annotation_cols)}. "
                f"These will be shown during annotation. "
                f"{unannotated_count} rows remain unannotated."
            )

    # If no columns selected, or zero matches, we still return everything so the app doesn't break
    return df, selected_annotation_cols, annotated_count, unannotated_count, total_count
