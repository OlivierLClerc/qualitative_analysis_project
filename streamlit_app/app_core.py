"""
Main module for the Streamlit app.
This module imports and uses all the other modules to create the complete app.
"""

import streamlit as st
from typing import Any, Dict, List, Optional
import pandas as pd

from streamlit_app.data_upload import upload_dataset
from streamlit_app.session_management import load_previous_session, save_session
from streamlit_app.column_selection import select_rename_describe_columns
from streamlit_app.codebook_management import codebook_and_examples
from streamlit_app.field_selection import select_fields
from streamlit_app.llm_configuration import configure_llm
from streamlit_app.analysis import run_analysis
from streamlit_app.evaluation import compare_with_external_judgments


class QualitativeAnalysisApp:
    def __init__(self) -> None:
        """
        Initializes the QualitativeAnalysisApp by pulling default or stored values
        from Streamlit's session_state.
        """
        self.data: Optional[pd.DataFrame] = st.session_state.get("data", None)
        self.processed_data: Optional[pd.DataFrame] = st.session_state.get(
            "processed_data", None
        )

        # Keep track of which columns are annotation columns
        self.annotation_columns: List[str] = st.session_state.get(
            "annotation_columns", []
        )

        self.selected_columns: List[str] = st.session_state.get("selected_columns", [])
        self.column_renames: Dict[str, str] = st.session_state.get("column_renames", {})
        self.column_descriptions: Dict[str, str] = st.session_state.get(
            "column_descriptions", {}
        )

        self.codebook: str = st.session_state.get("codebook", "")
        self.examples: str = st.session_state.get("examples", "")

        self.llm_client: Any = None  # Will instantiate later (OpenAI or Together)
        self.selected_model: Optional[str] = st.session_state.get(
            "selected_model", None
        )

        self.selected_fields: List[str] = st.session_state.get("selected_fields", [])
        self.results: List[Dict[str, Any]] = st.session_state.get("results", [])

    def run(self) -> None:
        """
        Main entry point for the Streamlit app.
        Executes each analysis step in sequence if the required data is available.
        """
        st.title("Qualitative Analysis")

        # App Purpose Explanation
        st.markdown(
            """
            ### About This App
            This **Qualitative Analysis App** helps you analyze qualitative datasets 
            using Large Language Models.
            You will need a dataset to analyse, a codebook, and a valid API key (OpenAi or Together).
            """,
            unsafe_allow_html=True,
        )

        # Step 1: Upload Dataset
        self.data = upload_dataset(self.data, st.session_state)

        # Steps 2-7 are only relevant if data is uploaded
        if self.data is not None:
            # Load previous session if a JSON is imported
            load_previous_session(self)

            # Step 2: Select & Rename Columns, Add Descriptions, plus annotation filtering
            self.processed_data = select_rename_describe_columns(self, self.data)

            # Step 3: Codebook & Examples
            self.codebook, self.examples = codebook_and_examples(self)

            # Step 4: Fields to Extract
            self.selected_fields = select_fields(self)

            # Offer to Save Session here before moving to Step 5
            save_session(self)

            # Step 5: Configure LLM (provider & model)
            self.llm_client = configure_llm(self)

            # Step 6: Run Analysis
            # results_df = run_analysis(self)
            run_analysis(self)

            # Step 7: Compare with External Judgments (optionally: alt-test or Cohen's Kappa)
            compare_with_external_judgments(self)


def main() -> None:
    app = QualitativeAnalysisApp()
    app.run()


if __name__ == "__main__":
    main()
