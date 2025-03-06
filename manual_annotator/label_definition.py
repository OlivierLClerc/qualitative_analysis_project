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
        """
        Define the possible labels of annotations that you can apply. 

        For example, if you are annotating text and are interested in sentiment analysis, 
        you might use labels like 'positive', 'negative', and 'neutral'. 
        The labels will be used to create buttons for annotating the data. 
        
        Simply type your different labels in a comma-separated list (e.g. 'positive, negative, neutral').
        """
    )

    # Text input for labels
    updated_labels = st.text_input(
        "Labels (comma-separated):",
        value=fast_labels_text,
        key="fast_labels_text",
        placeholder="e.g. 0, 1, or cat, dog, etc.",
    )

    # Button to confirm label application
    if st.button("Apply Labels"):
        st.success("Labels applied successfully!")

    return updated_labels
