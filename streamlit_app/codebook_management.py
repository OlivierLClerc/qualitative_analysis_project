"""
Module for handling codebook and examples functionality in the Streamlit app.
"""

import streamlit as st
from typing import Any, Tuple


def codebook_and_examples(app_instance: Any) -> Tuple[str, str]:
    """
    Step 3: Codebook & Examples
    Lets the user define or modify the classification codebook and (optionally) examples
    that guide the LLM in producing structured responses.

    Args:
        app_instance: The QualitativeAnalysisApp instance

    Returns:
        A tuple containing the codebook and examples
    """
    st.markdown("### Step 3: Codebook & Examples", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 3", expanded=True):
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
