"""
Module for handling dataset upload functionality in the Streamlit app.
"""

import streamlit as st
from typing import Optional, Any, Dict, Union
import pandas as pd
import json

from qualitative_analysis import load_data
from streamlit_app.session_management import load_previous_session
from streamlit_app.gameplay_utils import (
    is_gameplay_config,
    get_available_gameplays,
    validate_gameplay_in_data,
    get_gameplay_config,
    validate_columns_exist,
    filter_df_by_gameplay,
    get_gameplay_description,
)


def upload_dataset(
    app_instance: Any,
    session_state: Union[Dict[Any, Any], "st.runtime.state.SessionStateProxy"],
) -> Optional[pd.DataFrame]:
    """
    Step 1: Uploads a dataset (CSV or XLSX) via Streamlit's file uploader.
    - Validates file type and delimiter.
    - Loads data into data and session_state.

    Args:
        app_instance: The QualitativeAnalysisApp instance
        session_state: Streamlit's session state dictionary

    Returns:
        The loaded DataFrame or None if no file was uploaded
    """

    st.markdown("### Step 1: Upload Your Dataset", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 1", expanded=True):
        # Expected Data Format Explanation
        st.markdown(
            """
            ### Expected Data Format
            Your dataset should be in **CSV** or **Excel (XLSX)** format.

            **Required Structure:**  
            - Each row must have a **unique ID** and represent a single entry.  
            - The dataset should contain at least three columns:  
                - One for the **unique identifier**  
                - One or more **data columns** containing text or information to analyze.  
                - One or more **annotation columns** with human judgments or labels. 
                Those columns will be used to compare the model's predictions and determine if the model can be used on the rest of the data.
                Therefore, you only neeed annotations on a subset of the data.
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

        if uploaded_file is not None:
            file_type = "csv" if uploaded_file.name.endswith(".csv") else "xlsx"
            delimiter = st.text_input("CSV Delimiter (if CSV)", value=";")

            try:
                data = load_data(
                    uploaded_file, file_type=file_type, delimiter=delimiter
                )
                # Reset session states relevant to data
                session_state["selected_columns"] = []
                session_state["column_renames"] = {}
                session_state["column_descriptions"] = {}
                session_state["annotation_columns"] = []

                st.success("Data loaded successfully!")
                st.write("Data Preview:", data.head())

                # Store in session_state
                session_state["data"] = data
                app_instance.data = data

            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()

            # NEW: Gameplay mode option
            st.markdown("---")
            st.markdown("### Gameplay Mode (Optional)")
            use_gameplay = st.checkbox(
                "Use gameplay template configuration?",
                value=session_state.get("use_gameplay_mode", False),
                key="use_gameplay_checkbox",
                help="Enable this to use a gameplay-based configuration template that auto-populates column descriptions and codebooks",
            )

            session_state["use_gameplay_mode"] = use_gameplay

            if use_gameplay:
                st.markdown(
                    """
                Upload a **gameplay configuration JSON** that contains:
                - Common columns for all gameplays
                - Gameplay-specific columns and codebooks
                
                The configuration should have a `"gameplays"` section with your gameplay types.
                """
                )

                gameplay_config_file = st.file_uploader(
                    "Upload Gameplay Config (JSON)",
                    type=["json"],
                    key="gameplay_config_uploader",
                )

                if gameplay_config_file is not None:
                    try:
                        gameplay_config = json.load(gameplay_config_file)

                        # Validate it's a gameplay config
                        if not is_gameplay_config(gameplay_config):
                            st.error(
                                "❌ This JSON doesn't contain a 'gameplays' section. Please use a gameplay configuration file."
                            )
                            return None

                        # Store in session
                        session_state["gameplay_config"] = gameplay_config

                        # Get available gameplays
                        available_gameplays = get_available_gameplays(gameplay_config)

                        st.success(
                            f"✅ Gameplay config loaded! Available gameplays: {', '.join(available_gameplays)}"
                        )

                        # Let user select gameplay
                        st.markdown("### Select Gameplay Type")

                        # Get previously selected gameplay if any
                        default_index = 0
                        if (
                            "selected_gameplay" in session_state
                            and session_state["selected_gameplay"]
                            in available_gameplays
                        ):
                            default_index = available_gameplays.index(
                                session_state["selected_gameplay"]
                            )

                        selected_gameplay = st.selectbox(
                            "Choose which gameplay type to analyze:",
                            options=available_gameplays,
                            index=default_index,
                            key="gameplay_selector",
                        )

                        # Show gameplay description
                        gameplay_desc = get_gameplay_description(
                            gameplay_config, selected_gameplay
                        )
                        if gameplay_desc:
                            st.info(f"ℹ️ **{selected_gameplay}**: {gameplay_desc}")

                        # Validate gameplay exists in data
                        is_valid, error_msg = validate_gameplay_in_data(
                            data, selected_gameplay
                        )
                        if not is_valid:
                            st.error(f"❌ {error_msg}")
                            return None

                        # Get merged config for this gameplay
                        merged_config = get_gameplay_config(
                            gameplay_config, selected_gameplay
                        )

                        # Validate required columns exist
                        required_cols = list(
                            merged_config["column_descriptions"].keys()
                        )
                        all_exist, missing = validate_columns_exist(data, required_cols)
                        if not all_exist:
                            st.error(
                                f"❌ Missing required columns in dataset: {', '.join(missing)}"
                            )
                            st.info(f"💡 Required columns: {', '.join(required_cols)}")
                            return None

                        # Filter data by gameplay
                        filtered_data = filter_df_by_gameplay(data, selected_gameplay)
                        st.success(
                            f"✅ Filtered to **{len(filtered_data)}** rows for gameplay: **{selected_gameplay}**"
                        )

                        # Store everything
                        session_state["selected_gameplay"] = selected_gameplay
                        session_state["gameplay_merged_config"] = merged_config
                        session_state["data"] = filtered_data
                        app_instance.data = filtered_data

                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON file. Please check the file format.")
                        return None
                    except Exception as e:
                        st.error(f"❌ Error loading gameplay config: {e}")
                        return None
                else:
                    st.info(
                        "Upload a gameplay configuration JSON to continue with gameplay mode"
                    )
                    return None
            else:
                # Standard mode - load existing session if provided
                load_previous_session(app_instance)
        else:
            # No file uploaded yet, show load previous session option
            load_previous_session(app_instance)

    return app_instance.data
