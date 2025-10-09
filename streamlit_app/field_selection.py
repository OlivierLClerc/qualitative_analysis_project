"""
Module for handling field selection functionality in the Streamlit app.
"""

import streamlit as st
from typing import Any, List
from streamlit_app.session_management import save_session


def select_fields(app_instance: Any, step_number: int = 4) -> List[str]:
    """
    Fields to Extract
    Allows the user to specify which fields (e.g., 'Evaluation', 'Comments')
    the LLM should return in its JSON output.
    Also asks which field is the label column.

    Args:
        app_instance: The QualitativeAnalysisApp instance
        step_number: The step number to display (default=4 for annotation mode)

    Returns:
        A list of selected fields
    """
    st.markdown(f"### Step {step_number}: Fields to Extract", unsafe_allow_html=True)
    with st.expander(f"Show/hide details of step {step_number}", expanded=True):
        st.markdown(
            """
            Specify the **fields** (or categories) you want the model to generate for each entry.
            The names should match the field names used in your codebook & examples (if provided).
            """,
            unsafe_allow_html=True,
        )

        default_fields = (
            ",".join(app_instance.selected_fields)
            if app_instance.selected_fields
            else ""
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

            # Get the default index for the label column
            default_index = 0
            if (
                app_instance.label_column
                and app_instance.label_column in app_instance.selected_fields
            ):
                default_index = app_instance.selected_fields.index(
                    app_instance.label_column
                )
            elif not app_instance.selected_fields:
                default_index = None

            # Select the label column from the extracted fields
            label_column = st.selectbox(
                "Label Column:",
                options=app_instance.selected_fields,
                index=default_index,
                key="label_column_select",
            )

            # Store in session state and app instance
            st.session_state["label_column"] = label_column
            app_instance.label_column = label_column

        # Add the save session functionality at the end of Step 4
        save_session(app_instance)

    return extracted
