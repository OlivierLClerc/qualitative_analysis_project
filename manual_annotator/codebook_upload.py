"""
Module for handling codebook upload functionality in the Manual Annotation Tool.
"""

import streamlit as st


def load_codebook(file) -> str:
    """
    Loads the codebook text from a TXT file and returns it as a single string.

    Args:
        file: The uploaded file object

    Returns:
        The codebook text as a string
    """
    return file.read().decode("utf-8")


def upload_codebook(codebook_text: str) -> str:
    """
    Step 4: Upload Codebook
    Allows the user to upload a TXT file with classification instructions.

    Args:
        codebook_text: Current codebook text

    Returns:
        The updated codebook text
    """
    st.header("Step 4: Upload Codebook (Optional)")

    st.markdown(
        "You may upload a TXT file with classification instructions. "
        "Its contents will be displayed on the sidebar while you annotate."
    )

    codebook_file = st.file_uploader(
        "Upload Codebook TXT file:", type=["txt"], key="codebook_file"
    )

    if codebook_file is not None:
        codebook_str = load_codebook(codebook_file)
        codebook_text = codebook_str
        st.session_state.codebook_text = codebook_text
        st.success("Codebook loaded successfully!")

    # If a codebook has been loaded, show it in the sidebar
    if codebook_text:
        st.sidebar.header("Codebook Instructions")
        formatted_text = codebook_text.replace("\n", "<br>")
        st.sidebar.markdown(formatted_text, unsafe_allow_html=True)

    return codebook_text
