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

    # Use a stable key for the multiselect, and provide a default from session state
    selected_annotation_cols: List[str] = st.multiselect(
        label="Select column(s) that contain existing human annotations:",
        options=df.columns,
        default=st.session_state.get("selected_annotation_cols", []),
        help="Only rows with non-null values in ALL selected columns will be shown during annotation.",
        key="annotation_cols_key",  # Unique key
    )

    # Update our session state copy of the selection
    st.session_state["selected_annotation_cols"] = selected_annotation_cols

    # Initialize variables
    annotated_indices = []
    annotated_count = 0
    unannotated_count = 0
    total_count = len(df)

    # Identify annotated rows if annotation columns are selected
    if selected_annotation_cols:
        # Get indices of rows WITH annotations
        annotated_mask = df[selected_annotation_cols].notna().all(axis=1)
        annotated_indices = df[annotated_mask].index.tolist()

        # Calculate counts
        annotated_count = len(annotated_indices)
        unannotated_count = total_count - annotated_count

        # Store in session state
        st.session_state.annotated_indices = annotated_indices
        st.session_state.annotated_count = annotated_count
        st.session_state.unannotated_count = unannotated_count
        st.session_state.total_count = total_count

        st.success(
            f"Identified {annotated_count} rows with non-null values in {', '.join(selected_annotation_cols)}. "
            f"These rows will be shown during annotation. "
            f"{unannotated_count} more unannotated rows will be included in final download."
        )

        if annotated_count == 0:
            st.warning("No rows with annotations found! Please adjust your filters.")
            st.stop()
            return df, selected_annotation_cols, 0, 0, 0

        return (
            df,
            selected_annotation_cols,
            annotated_count,
            unannotated_count,
            total_count,
        )

    # If no annotation columns selected, all rows are considered "annotated" for display purposes
    return df, selected_annotation_cols, total_count, 0, total_count
