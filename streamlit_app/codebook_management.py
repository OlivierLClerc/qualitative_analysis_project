"""
Module for handling codebook and examples functionality in the Streamlit app.
"""

import streamlit as st
from typing import Any, Tuple


def codebook_and_examples(app_instance: Any, step_number: int = 3) -> Tuple[str, str]:
    """
    Codebook & Examples
    Lets the user define or modify the classification codebook and (optionally) examples
    that guide the LLM in producing structured responses.

    Args:
        app_instance: The QualitativeAnalysisApp instance
        step_number: The step number to display (default=3 for annotation mode)

    Returns:
        A tuple containing the codebook and examples
    """
    st.markdown(f"### Step {step_number}: Codebook & Examples", unsafe_allow_html=True)
    with st.expander(f"Show/hide details of step {step_number}", expanded=True):
        # Guidance on how to write the codebook and examples
        st.markdown(
            """
            ### **Codebook Instructions**
            Define how the data should be analyzed (as you would do for a human annotator).
            """,
            unsafe_allow_html=True,
        )

        default_codebook = st.session_state.get("codebook", "")
        default_examples = st.session_state.get("examples", "")

        codebook_val = st.text_area(
            "**Your Codebook / Instructions for LLM:**",
            value=default_codebook,
            key="codebook_textarea",
            height=400,
        )
        examples_val = st.text_area(
            "**Your Examples (Optional):**",
            value=default_examples,
            key="examples_textarea",
            height=400,
        )

        # Update app instance and session state
        app_instance.codebook = codebook_val
        app_instance.examples = examples_val

        st.session_state["codebook"] = codebook_val
        st.session_state["examples"] = examples_val

    return codebook_val, examples_val
