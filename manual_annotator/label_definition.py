"""
Module for handling label definition functionality in the Manual Annotation Tool.
"""

import streamlit as st


def define_labels(fast_labels_text: str) -> str:
    """
    Step 5: Define Labels
    Allows the user to define the possible labels of annotations.

    Args:
        fast_labels_text: Current comma-separated labels

    Returns:
        The updated comma-separated labels
    """
    st.header("Step 5: Define Labels")

    st.markdown(
        "Define the possible labels of annotations that you can apply. "
        "Simply type them in a comma-separated list."
    )

    updated_labels = st.text_input(
        "Labels (comma-separated):",
        value=fast_labels_text,
        key="fast_labels_text",
        placeholder="e.g. 0, 1, or cat, dog, etc.",
    )

    # Don't update session state here - Streamlit does this automatically
    # when using the key parameter

    return updated_labels
