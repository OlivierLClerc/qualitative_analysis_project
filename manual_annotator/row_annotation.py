"""
Module for handling row annotation functionality in the Manual Annotation Tool.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple
from transformers import pipeline


@st.cache_resource(show_spinner=False)
def get_translator():
    """
    Translator pipeline for FR -> EN.

    Returns:
        A translation pipeline
    """
    return pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")


def annotate_rows(
    df: pd.DataFrame,
    current_index: int,
    selected_columns: List[str],
    new_col_name: str,
    annotator_name: str,
    fast_labels_text: str,
    fast_label: str,
    translated_rows: Dict[int, Dict[str, str]],
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

    Returns:
        A tuple containing:
        - df: The updated DataFrame
        - current_index: The updated row index
        - fast_label: The updated selected label
        - translated_rows: The updated dictionary of translated rows
    """
    st.header("Step 7: Annotate Rows")

    # Get annotated indices if available
    annotated_indices = st.session_state.get("annotated_indices", [])

    # If we have annotation columns and annotated indices, use them to navigate
    if (
        annotated_indices
        and "selected_annotation_cols" in st.session_state
        and st.session_state["selected_annotation_cols"]
    ):
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

        st.info(
            f"Showing annotated row {current_index + 1} of {len(annotated_indices)}"
        )
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
    flag_col = f"Unvalid_{annotator_name}"

    # Show existing rating & flagged status
    rating_val = df.at[idx, new_col_name]
    flagged_val = df.at[idx, flag_col] if flag_col in df.columns else None

    st.markdown(f"**Row Index:** {idx}")
    st.markdown(f"**Existing Rating:** {rating_val}")
    st.markdown(f"**Is Unvalid:** {flagged_val}")

    # Display the selected columns
    for col in selected_columns:
        val = df.at[idx, col]

        if pd.notna(val):
            # If it's a float and has no decimal part, convert to int for display
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

            # Update session state
            st.session_state.translated_rows = translated_rows

        # Display the stored translation
        translations = translated_rows[idx]
        st.markdown("### Translated Content:")
        for col, tval in translations.items():
            st.write(f"**{col}:** {tval}")

    # Navigation Buttons
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Previous"):
            # Apply the currently selected fast label if any
            if st.session_state.fast_label != "":
                df.at[idx, new_col_name] = st.session_state.fast_label

            # Reset the fast label in session state
            st.session_state.fast_label = ""

            # Handle navigation (with annotated indices if applicable)
            if annotated_indices and st.session_state.get(
                "selected_annotation_cols", []
            ):
                current_index = max(0, current_index - 1)
            else:
                current_index = max(0, current_index - 1)

            st.session_state.current_index = current_index
            st.rerun()

    with c2:
        if st.button("Next"):
            if st.session_state.fast_label != "":
                df.at[idx, new_col_name] = st.session_state.fast_label
            st.session_state.fast_label = ""
            if annotated_indices and st.session_state.get(
                "selected_annotation_cols", []
            ):
                current_index = min(len(annotated_indices) - 1, current_index + 1)
            else:
                current_index = min(len(df) - 1, current_index + 1)
            st.session_state.current_index = current_index
            st.rerun()

    with c3:
        if st.button("Unvalid data"):
            df.at[idx, flag_col] = True
            if st.session_state.fast_label != "":
                df.at[idx, new_col_name] = st.session_state.fast_label
            st.session_state.fast_label = ""
            if annotated_indices and st.session_state.get(
                "selected_annotation_cols", []
            ):
                current_index = min(len(annotated_indices) - 1, current_index + 1)
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
