"""
Module for handling field selection functionality in the Streamlit app.
"""

import streamlit as st
from typing import Any, List


def select_fields(app_instance: Any) -> List[str]:
    """
    Step 4: Fields to Extract
    Allows the user to specify which fields (e.g., 'Evaluation', 'Comments')
    the LLM should return in its JSON output.
    Also asks which field is the label column.

    Args:
        app_instance: The QualitativeAnalysisApp instance

    Returns:
        A list of selected fields
    """
    st.header("Step 4: Fields to Extract")

    st.markdown(
        """
        Specify the **fields** (or categories) you want the model to generate for each entry.
        The names should match the field names used in your codebook & examples (if provided).
        """,
        unsafe_allow_html=True,
    )

    default_fields = (
        ",".join(app_instance.selected_fields) if app_instance.selected_fields else ""
    )
    fields_str = st.text_input(
        "Comma-separated fields (e.g. 'Reasoning, Classification')",
        value=default_fields,
        key="fields_input",
    )
    extracted = [f.strip() for f in fields_str.split(",") if f.strip()]

    # Update app instance and session state
    app_instance.selected_fields = extracted
    st.session_state["selected_fields"] = extracted

    # Only show label selection if fields have been specified and annotation columns exist
    if app_instance.selected_fields and app_instance.annotation_columns:
        st.subheader("Label Column Selection")
        st.markdown(
            """
            Select which field is your **label column** (the main classification or prediction)
            that should match the format of your annotation columns.
            """
        )

        # Select the label column from the extracted fields
        label_column = st.selectbox(
            "Label Column:",
            options=app_instance.selected_fields,
            index=0 if app_instance.selected_fields else None,
            key="label_column_select",
        )

        # Store in session state
        st.session_state["label_column"] = label_column

    return extracted
