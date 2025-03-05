"""
Main entry point for the Manual Annotation Tool.
"""

from manual_annotator import ManualAnnotatorApp
import streamlit as st


if __name__ == "__main__":
    # Inject custom CSS to adjust layout and component spacing
    st.markdown(
        """
        <style>
        /* Adjust the main container margins and max-width */
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 1200px;
        }
        /* Force the radiogroup container to display radio options horizontally */
        div[role="radiogroup"] {
            display: flex;
            flex-direction: row;
            margin-bottom: 0.1rem;
        }
        /* Reduce spacing between radio options */
        div[role="radiogroup"] label {
            margin-right: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create and run the app
    app = ManualAnnotatorApp()
    app.run()
