"""
Main entry point for the Qualitative Analysis Streamlit app.
This file imports the modularized app from the streamlit_app package.
"""

from streamlit_app import QualitativeAnalysisApp


def main() -> None:
    """
    Main function that initializes and runs the Qualitative Analysis app.
    """
    app = QualitativeAnalysisApp()
    app.run()


if __name__ == "__main__":
    main()
