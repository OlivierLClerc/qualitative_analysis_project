"""
Wrapper for rendering evaluation metrics inside annotation mode.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from streamlit_app.evaluation_dashboard import render_evaluation_dashboard
from streamlit_app.evaluation_mappings import sanitize_evaluation_mappings


ANNOTATION_EVALUATION_PREFIX = "annotation_"


def compare_with_external_judgments(app_instance: Any) -> None:
    """
    Step 7: Compare LLM outputs with external judgments.
    """
    st.markdown("### Step 7: Evaluate Model Performance", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 7", expanded=True):
        if not app_instance.results:
            st.warning("No analysis results. Please run the analysis first.")
            return

        if not app_instance.annotation_columns:
            st.info("No annotation columns were selected in Step 2.")
            return

        results_df: pd.DataFrame = st.session_state["results_df"]
        evaluation_mappings = sanitize_evaluation_mappings(
            raw_mappings=getattr(app_instance, "evaluation_mappings", []),
            selected_fields=app_instance.selected_fields,
            annotation_columns=app_instance.annotation_columns,
            legacy_label_column=app_instance.label_column,
            legacy_label_type=app_instance.label_type,
            create_default_if_empty=False,
        )
        app_instance.evaluation_mappings = evaluation_mappings
        st.session_state["evaluation_mappings"] = evaluation_mappings

        if not evaluation_mappings:
            st.info("Add at least one evaluation mapping in Step 4 to compute metrics.")
            return

        render_evaluation_dashboard(
            results_df=results_df,
            evaluation_mappings=evaluation_mappings,
            ui_key_prefix=ANNOTATION_EVALUATION_PREFIX,
            comparison_label="LLM field",
            comparison_entity_name="LLM",
            intro_markdown="""
This step measures how closely your LLM outputs align with existing human annotations.
Each evaluation mapping is computed independently, so you can compare several dimensions
such as clarity, creativity, or validity in the same run.

We provide four comparison methods:
- **Cohen's Kappa** for annotator agreement
- **Classification Metrics** for accuracy and per-class performance
- **Alt-Test** for model viability against human annotators
- **Krippendorff's Alpha** for non-inferiority analysis
""",
        )
