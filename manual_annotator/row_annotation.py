"""
Module for handling row annotation functionality in the Manual Annotation Tool.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple, Optional
from transformers import pipeline


@st.cache_resource(show_spinner=False)
def get_translator():
    """
    Translator pipeline for FR -> EN.

    Returns:
        A translation pipeline
    """
    return pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")


def is_valid_annotated_row(
    df: pd.DataFrame, idx: int, annotation_cols: List[str]
) -> bool:
    """
    Check if a row has non-null values in all of the selected annotation columns.

    Args:
        df: The DataFrame to check
        idx: The index of the row to check
        annotation_cols: List of column names that contain annotations

    Returns:
        True if the row has non-null values in all of the selected annotation columns, False otherwise
    """
    if not annotation_cols:
        return True

    for col in annotation_cols:
        if pd.isna(df.at[idx, col]):
            return False

    return True


def annotate_rows(
    df: pd.DataFrame,
    current_index: int,
    selected_columns: List[str],
    new_col_name: str,
    annotator_name: str,
    fast_labels_text: str,
    fast_label: str,
    translated_rows: Dict[int, Dict[str, str]],
    sort_column: Optional[str] = None,
    enable_sorting: bool = False,
) -> Tuple[pd.DataFrame, int, str, Dict[int, Dict[str, str]]]:
    """
    Step 7: Annotate Row-by-Row
    Allows the user to annotate each row one by one.

    Args:
        df: The DataFrame to annotate
        current_index: The current row index
        selected_columns: List of columns to display
        new_col_name: The column name for annotations
        annotator_name: The annotator name
        fast_labels_text: Comma-separated labels
        fast_label: Currently selected label
        translated_rows: Dictionary of translated rows
        sort_column: Optional column name to sort by
        enable_sorting: Whether sorting is enabled

    Returns:
        A tuple containing:
        - df: The updated DataFrame
        - current_index: The updated row index
        - fast_label: The updated selected label
        - translated_rows: The updated dictionary of translated rows
    """
    st.header("Step 7: Annotate Rows")

    # Initialize fast_label in session state if not present
    if "fast_label" not in st.session_state:
        st.session_state.fast_label = fast_label

    # Apply sorting if enabled
    if enable_sorting and sort_column and sort_column in df.columns:
        # Create a sorted copy of the dataframe
        sorted_df = df.sort_values(by=sort_column)
        # Get the sorted indices
        sorted_indices = sorted_df.index.tolist()
        # Store the sorted indices in session state
        st.session_state.sorted_indices = sorted_indices
    else:
        # Clear sorted indices if sorting is disabled
        if "sorted_indices" in st.session_state:
            del st.session_state.sorted_indices

    # Get annotated indices if available (these are the filtered rows from Step 2)
    annotated_indices = st.session_state.get("annotated_indices", [])

    # Get selected annotation columns
    selected_annotation_cols = st.session_state.get("selected_annotation_cols", [])

    # Store original filtered indices before potentially overriding with sorted indices
    filtered_indices = list(annotated_indices) if annotated_indices else []

    # If sorting is enabled and we have sorted indices, use them for navigation
    if enable_sorting and sort_column and "sorted_indices" in st.session_state:
        # Always use sorted indices when sorting is enabled, regardless of filtering
        annotated_indices = st.session_state.sorted_indices

    # If we have indices to use for navigation (either filtered or sorted)
    if annotated_indices:
        # Convert current_index to actual dataframe index if we're using annotated indices
        if current_index >= len(annotated_indices):
            current_index = len(annotated_indices) - 1

        if current_index < 0:
            current_index = 0

        # Get the actual dataframe index
        if annotated_indices:
            idx = annotated_indices[current_index]
        else:
            idx = 0

        # Verify that the row has non-null values in all selected annotation columns
        # Only check this if we're using filtered indices
        if selected_annotation_cols and not is_valid_annotated_row(
            df, idx, selected_annotation_cols
        ):
            # This shouldn't happen if the annotated_indices are correct, but just in case
            st.warning(
                f"Row {idx} does not have values in all selected annotation columns. "
                f"This may indicate an issue with the filtering."
            )

        # Show appropriate message based on whether we're using sorted or filtered indices
        if enable_sorting and sort_column and not selected_annotation_cols:
            st.info(
                f"Showing sorted row {current_index + 1} of {len(annotated_indices)} (sorted by {sort_column})"
            )
        else:
            # Use filtered_indices.length for the total count if available
            total_count = (
                len(filtered_indices) if filtered_indices else len(annotated_indices)
            )
            st.info(f"Showing annotated row {current_index + 1} of {total_count}")
    else:
        # No filtering, use regular indices
        idx = current_index
        # Ensure index is in valid range
        if idx < 0:
            current_index = 0
            idx = 0
        if idx >= len(df):
            current_index = len(df) - 1
            idx = len(df) - 1

    # Update session state
    st.session_state.current_index = current_index

    # Get flag column name
    flag_col = f"Invalid_{annotator_name}"

    # Show existing rating & flagged status
    rating_val = df.at[idx, new_col_name]
    if (
        pd.notna(rating_val)
        and isinstance(rating_val, float)
        and rating_val.is_integer()
    ):
        rating_val = int(rating_val)
    flagged_val = df.at[idx, flag_col] if flag_col in df.columns else None

    # Count remaining unannotated rows in the filtered subset
    remaining_unannotated = 0
    total_in_subset = 0

    # Use filtered_indices for counting, not annotated_indices which might be overridden by sorting
    if filtered_indices:
        # Get the total number of rows in the filtered subset
        total_in_subset = len(filtered_indices)

        # Count how many rows in the filtered subset have been annotated
        annotated_count = 0
        for index in filtered_indices:
            if pd.notna(df.at[index, new_col_name]):
                annotated_count += 1

        # Calculate remaining unannotated rows
        remaining_unannotated = total_in_subset - annotated_count
    else:
        # If no filtering, count all unannotated rows
        remaining_unannotated = df[pd.isna(df[new_col_name])].shape[0]

    st.markdown(f"**Row Index:** {idx}")
    st.markdown(f"**Your current label:** {rating_val}")
    st.markdown(f"**Is Invalid:** {flagged_val}")
    st.markdown(f"**Remaining unannotated data:** {remaining_unannotated}")

    # Display the selected columns
    for col in selected_columns:
        val = df.at[idx, col]
        if pd.notna(val):
            if isinstance(val, float) and val.is_integer():
                val = int(val)
        st.write(f"**{col}:** {val}")

    # Translation (optional)
    translate_row = st.checkbox("Translate this row to English", key="translate_row")
    if translate_row:
        if idx not in translated_rows:
            translator = get_translator()
            translation_dict = {}
            for col in selected_columns:
                text = str(df.at[idx, col])
                try:
                    result = translator(text)
                    translation_dict[col] = result[0]["translation_text"]
                except Exception as e:
                    st.error(f"Error translating '{col}': {e}")
                    translation_dict[col] = "[Error]"
            translated_rows[idx] = translation_dict
            st.session_state.translated_rows = translated_rows

        translations = translated_rows[idx]
        st.markdown("### Translated Content:")
        for col, tval in translations.items():
            st.write(f"**{col}:** {tval}")

    # Navigation Buttons
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("Previous"):
            if st.session_state.fast_label != "":
                df.at[idx, new_col_name] = st.session_state.fast_label
            st.session_state.fast_label = ""

            if annotated_indices:
                # Find the previous valid index
                new_index = current_index - 1
                while new_index >= 0:
                    candidate_idx = annotated_indices[new_index]
                    # Only check validity if we have selected annotation columns
                    if not selected_annotation_cols or is_valid_annotated_row(
                        df, candidate_idx, selected_annotation_cols
                    ):
                        current_index = new_index
                        break
                    new_index -= 1

                # If no valid index found, stay at the current index
                if new_index < 0:
                    current_index = max(0, current_index)
            else:
                current_index = max(0, current_index - 1)

            st.session_state.current_index = current_index
            st.rerun()

    with c2:
        if st.button("Next"):
            if st.session_state.fast_label != "":
                df.at[idx, new_col_name] = st.session_state.fast_label
            st.session_state.fast_label = ""

            if annotated_indices:
                # Find the next valid index
                new_index = current_index + 1
                while new_index < len(annotated_indices):
                    candidate_idx = annotated_indices[new_index]
                    # Only check validity if we have selected annotation columns
                    if not selected_annotation_cols or is_valid_annotated_row(
                        df, candidate_idx, selected_annotation_cols
                    ):
                        current_index = new_index
                        break
                    new_index += 1

                # If no valid index found, stay at the current index
                if new_index >= len(annotated_indices):
                    current_index = min(len(annotated_indices) - 1, current_index)
            else:
                current_index = min(len(df) - 1, current_index + 1)

            st.session_state.current_index = current_index
            st.rerun()

    with c3:
        if st.button("Next unrated"):
            if st.session_state.fast_label != "":
                df.at[idx, new_col_name] = st.session_state.fast_label
            st.session_state.fast_label = ""
            found = False

            if annotated_indices:
                for offset in range(current_index + 1, len(annotated_indices)):
                    candidate_idx = annotated_indices[offset]
                    # Check if the row is valid and unrated
                    # Only check validity if we have selected annotation columns
                    if (
                        not selected_annotation_cols
                        or is_valid_annotated_row(
                            df, candidate_idx, selected_annotation_cols
                        )
                    ) and pd.isna(df.at[candidate_idx, new_col_name]):
                        current_index = offset
                        found = True
                        break
            else:
                for i in range(current_index + 1, len(df)):
                    if pd.isna(df.at[i, new_col_name]):
                        current_index = i
                        found = True
                        break

            if found:
                st.session_state.current_index = current_index
                st.rerun()
            else:
                st.warning("No unrated rows found.")

    with c4:
        if st.button("Invalid data"):
            df.at[idx, flag_col] = True
            if st.session_state.fast_label != "":
                df.at[idx, new_col_name] = st.session_state.fast_label
            st.session_state.fast_label = ""

            if annotated_indices:
                # Find the next valid index
                new_index = current_index + 1
                while new_index < len(annotated_indices):
                    candidate_idx = annotated_indices[new_index]
                    # Only check validity if we have selected annotation columns
                    if not selected_annotation_cols or is_valid_annotated_row(
                        df, candidate_idx, selected_annotation_cols
                    ):
                        current_index = new_index
                        break
                    new_index += 1

                # If no valid index found, stay at the current index
                if new_index >= len(annotated_indices):
                    current_index = min(len(annotated_indices) - 1, current_index)
            else:
                current_index = min(len(df) - 1, current_index + 1)

            st.session_state.current_index = current_index
            st.rerun()

    # Fast Label (below data)
    fast_labels = [
        label.strip() for label in fast_labels_text.split(",") if label.strip()
    ]
    if fast_labels:
        selected_fast_label = st.radio(
            "Select a Label:", options=[""] + fast_labels, key="fast_label"
        )
        if selected_fast_label:
            st.markdown(f"**Current Label:** {selected_fast_label}")

    return df, current_index, fast_label, translated_rows
