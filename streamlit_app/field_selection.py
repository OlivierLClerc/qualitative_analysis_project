"""
Module for handling field selection and evaluation mapping configuration.
"""

from typing import Any, List

import streamlit as st

from streamlit_app.evaluation_mappings import (
    clear_evaluation_result_cache,
)
from streamlit_app.evaluation_mapping_editor import render_evaluation_mapping_editor
from streamlit_app.session_management import save_session


ANNOTATION_UI_PREFIX = "annotation_"


def select_fields(app_instance: Any, step_number: int = 4) -> List[str]:
    """
    Fields to Extract
    Allows the user to specify which fields (e.g., 'Evaluation', 'Comments')
    the LLM should return in its JSON output.
    Also asks which field is the label column when annotation columns are available.
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
            The names should match the field names used in your codebook and examples.
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "If you use the direct Gemini provider, field types and allowed values can be configured later "
            f"in Step {step_number + 1} > Advanced settings."
        )

        previous_fields = list(app_instance.selected_fields)
        default_fields = ",".join(previous_fields) if previous_fields else ""
        fields_str = st.text_input(
            "Comma-separated fields (e.g. 'Reasoning, Classification')",
            value=default_fields,
            key="fields_input",
        )
        extracted = [field.strip() for field in fields_str.split(",") if field.strip()]

        app_instance.selected_fields = extracted
        st.session_state["selected_fields"] = extracted

        # Only show label selection if fields have been specified and annotation columns exist
        if app_instance.selected_fields and app_instance.annotation_columns:
            st.subheader("Evaluation Mappings")
            updated_mappings = render_evaluation_mapping_editor(
                selected_fields=app_instance.selected_fields,
                annotation_columns=app_instance.annotation_columns,
                raw_mappings=st.session_state.get(
                    "evaluation_mappings", app_instance.evaluation_mappings
                ),
                legacy_label_column=app_instance.label_column,
                legacy_label_type=app_instance.label_type,
                mapping_state_key="evaluation_mappings",
                initialized_state_key="evaluation_mappings_initialized",
                ui_key_prefix=ANNOTATION_UI_PREFIX,
                llm_field_label="LLM field",
                intro_markdown="""
                Configure which LLM field should be evaluated against which human annotation columns.
                Each mapping is evaluated independently in Step 7.
                """,
            )

            mappings_changed = extracted != previous_fields or (
                updated_mappings != app_instance.evaluation_mappings
            )

            app_instance.evaluation_mappings = updated_mappings

            # Legacy config is no longer the runtime source of truth.
            app_instance.label_column = None
            app_instance.label_type = None
            st.session_state["label_column"] = None
            st.session_state["label_type"] = None

            if mappings_changed:
                clear_evaluation_result_cache(
                    st.session_state, prefix=ANNOTATION_UI_PREFIX
                )
        else:
            app_instance.evaluation_mappings = []
            st.session_state["evaluation_mappings"] = []
            st.session_state["evaluation_mappings_initialized"] = False
            clear_evaluation_result_cache(st.session_state, prefix=ANNOTATION_UI_PREFIX)

        save_session(app_instance)

    return extracted
