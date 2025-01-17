"""
app.py

A Streamlit application for qualitative data analysis. This app guides users
through dataset upload, cleaning, LLM-based classification, and optional 
comparison with external judgments.

Usage:
    streamlit run app.py
"""

from qualitative_analysis import (
    load_data,
    clean_and_normalize,
    sanitize_dataframe,
    build_data_format_description,
    construct_prompt,
    get_llm_client,
    parse_llm_response,
    compute_cohens_kappa,
    openai_api_calculate_cost,
)
import qualitative_analysis.config as config
import streamlit as st
import pandas as pd
from typing import Any, Dict, List, Optional
import io
import json


class QualitativeAnalysisApp:
    """
    A Streamlit-based application for qualitative data analysis.

    Attributes:
        data (pd.DataFrame | None): The original uploaded dataset.
        processed_data (pd.DataFrame | None): The dataset after cleaning and selected-column processing.
        selected_columns (List[str]): Columns chosen by the user.
        column_renames (Dict[str, str]): Mapping from original column names to renamed columns.
        column_descriptions (Dict[str, str]): Descriptions for each selected/renamed column.
        codebook (str): Instructions or guidelines for classification.
        examples (str): Sample examples or demonstrations for the LLM to follow.
        llm_client (Any | None): The instantiated LLM client (OpenAI or Together).
                                 (Type depends on your LLM client class.)
        selected_model (Optional[str]): The chosen model name (e.g., "gpt-4o").
        selected_fields (List[str]): Fields to extract from the LLM's response.
        results (List[Dict[str, Any]]): The final results of the analysis.
    """

    def __init__(self) -> None:
        """
        Initializes the QualitativeAnalysisApp by pulling default or stored values
        from Streamlit's session_state.
        """
        self.data: Optional[pd.DataFrame] = st.session_state.get("data", None)
        self.processed_data: Optional[pd.DataFrame] = st.session_state.get(
            "processed_data", None
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

        # Step 1: Upload Dataset
        self.upload_dataset()

        # Steps 2-4 are only relevant if data is uploaded
        if self.data is not None:
            # Load previous session if a JSON is imported
            self.load_previous_session()
            # Step 2: Select & Rename Columns, Add Descriptions
            self.select_rename_describe_columns()

            # Step 3: Codebook & Examples
            self.codebook_and_examples()

            # Step 4: Fields to Extract
            self.select_fields()

            # 💾 Offer to Save Session here before moving to Step 5
            self.save_session()

            # Step 5: Configure LLM (provider & model)
            self.configure_llm()

            # Step 6: Run Analysis
            self.run_analysis()

            # Step 7: Compare with External Judgments (Optional)
            self.compare_with_external_judgments()

    def upload_dataset(self) -> None:
        """
        Step 1: Uploads a dataset (CSV or XLSX) via Streamlit's file uploader.
        - Validates file type and delimiter.
        - Loads data into `self.data` and session_state.
        """
        st.header("Step 1: Upload Your Dataset")
        uploaded_file = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

        if uploaded_file is not None:
            file_type = "csv" if uploaded_file.name.endswith(".csv") else "xlsx"
            delimiter = st.text_input("CSV Delimiter (if CSV)", value=";")

            try:
                data = load_data(
                    uploaded_file, file_type=file_type, delimiter=delimiter
                )
                st.session_state["selected_columns"] = []
                st.session_state["column_renames"] = {}
                st.session_state["column_descriptions"] = {}

                st.success("Data loaded successfully!")
                st.write("Data Preview:", data.head())

                # Store in session_state + local attribute
                self.data = data
                st.session_state["data"] = data

            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()

    def load_previous_session(self) -> None:
        """
        Allows the user to upload a previous session configuration and restores the settings.
        """
        st.markdown(
            """
            <h4>🔄 <b>Load a Previous Session (Optional)</b></h4>
            <p style='font-size:16px'>
            If you've used this app before, you can upload your <b>saved session file (JSON)</b> to automatically restore previous settings
            (selected columns, codebook, examples, etc.).<br><br>
            🔍 <b>First time here?</b> After setting everything up, you'll have the option to <b>save your session</b> in <b>Step 4</b> 
            so you can easily continue next time without re-entering everything.
            </p>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload your saved session file (JSON):", type=["json"], key="load_session"
        )

        if uploaded_file is not None:
            try:
                # Load the JSON content directly
                session_data = json.load(uploaded_file)

                # Restore session values
                self.selected_columns = session_data.get("selected_columns", [])
                self.column_renames = session_data.get("column_renames", {})
                self.column_descriptions = session_data.get("column_descriptions", {})
                self.codebook = session_data.get("codebook", "")
                self.examples = session_data.get("examples", "")
                self.selected_fields = session_data.get("selected_fields", [])

                # Update session_state
                st.session_state["selected_columns"] = self.selected_columns
                st.session_state["column_renames"] = self.column_renames
                st.session_state["column_descriptions"] = self.column_descriptions
                st.session_state["codebook"] = self.codebook
                st.session_state["examples"] = self.examples
                st.session_state["selected_fields"] = self.selected_fields

                st.success("✅ Previous session successfully loaded!")

            except Exception as e:
                st.error(f"❌ Failed to load session: {e}")

    def select_rename_describe_columns(self) -> None:
        """
        Step 2: Lets the user select which columns to include, rename them,
        and provide descriptions. Also cleans and normalizes text columns.

        Stores processed data in `self.processed_data` and session_state.
        """
        st.header("Step 2: Select, Rename, and Describe Columns")

        if self.data is None:
            st.error("No dataset loaded.")
            return

        columns = self.data.columns.tolist()

        # Pull from session_state
        previous_selection = st.session_state.get("selected_columns", [])
        # Filter out invalid columns
        valid_previous_selection = [col for col in previous_selection if col in columns]

        st.write("Select which columns to include in your analysis:")
        self.selected_columns = st.multiselect(
            "Columns to include:",
            options=columns,
            default=valid_previous_selection if valid_previous_selection else columns,
        )
        st.session_state["selected_columns"] = self.selected_columns

        if not self.selected_columns:
            st.info("Select at least one column to proceed.")
            return

        # Rename columns
        for col in self.selected_columns:
            default_rename = self.column_renames.get(col, col)
            new_name = st.text_input(
                f"Rename '{col}' to:", value=default_rename, key=f"rename_{col}"
            )
            self.column_renames[col] = new_name
        st.session_state["column_renames"] = self.column_renames

        # Descriptions
        st.write("Add a short description for each selected column:")
        for col in self.selected_columns:
            col_key = self.column_renames[col]
            default_desc = self.column_descriptions.get(col_key, "")
            desc = st.text_area(
                f"Description for '{col_key}':",
                height=70,
                value=default_desc,
                key=f"desc_{col_key}",
            )
            self.column_descriptions[col_key] = desc
        st.session_state["column_descriptions"] = self.column_descriptions

        # Process & sanitize
        if self.selected_columns:
            processed = self.data[self.selected_columns].rename(
                columns=self.column_renames
            )

            text_cols: list[str] = st.multiselect(
                "Text columns:",
                processed.columns.tolist(),
                default=processed.columns.tolist(),
                key="text_columns_selection",
            )

            for tcol in text_cols:
                processed[tcol] = clean_and_normalize(processed[tcol])
            processed = sanitize_dataframe(processed)

            self.processed_data = processed
            st.session_state["processed_data"] = processed

            # Rebuild column_descriptions to only include renamed columns
            updated_column_descriptions: Dict[str, str] = {}
            for col in self.processed_data.columns:
                updated_column_descriptions[col] = self.column_descriptions.get(col, "")
            self.column_descriptions = updated_column_descriptions
            st.session_state["column_descriptions"] = self.column_descriptions

            st.success("Columns processed successfully!")
            st.write("Processed Data Preview:")
            st.dataframe(self.processed_data.head())

    def codebook_and_examples(self) -> None:
        """
        Step 3: Codebook & Examples
        Lets the user define or modify the classification codebook and (optionally) examples
        that guide the LLM in producing structured responses.
        """
        st.header("Step 3: Codebook & Examples")
        default_codebook = st.session_state.get("codebook", "")
        default_examples = st.session_state.get("examples", "")

        codebook_val = st.text_area(
            "Codebook / Instructions for LLM:",
            value=default_codebook,
            key="codebook_textarea",
        )
        examples_val = st.text_area(
            "Examples (Optional):", value=default_examples, key="examples_textarea"
        )

        self.codebook = codebook_val
        self.examples = examples_val

        st.session_state["codebook"] = codebook_val
        st.session_state["examples"] = examples_val

    def select_fields(self) -> None:
        """
        Step 4: Fields to Extract
        Allows the user to specify which fields (e.g., 'Evaluation', 'Comments')
        the LLM should return in its JSON output.
        """
        st.header("Step 4: Fields to Extract")
        default_fields = ",".join(self.selected_fields) if self.selected_fields else ""
        fields_str = st.text_input(
            "Comma-separated fields (e.g. 'Evaluation, Comments')",
            value=default_fields,
            key="fields_input",
        )
        extracted = [f.strip() for f in fields_str.split(",") if f.strip()]

        self.selected_fields = extracted
        st.session_state["selected_fields"] = extracted

    def save_session(self) -> None:
        """
        Allows the user to save the current session configuration (excluding API key).
        """

        # Custom Header (styled like "Load a Previous Session")
        st.markdown(
            """
            <h4><b>Save Your Session</b></h4>
            <p style='font-size:16px'>
            Save your current setup to avoid reconfiguring everything next time. <br><br>
            """,
            unsafe_allow_html=True,
        )

        # Let the user choose a filename
        filename_input = st.text_input(
            "**Enter a filename for your session:**",
            value="session_config.json",
            key="filename_input",
        )

        # Ensure filename ends with .json
        if not filename_input.endswith(".json"):
            filename_input += ".json"

        # Session data to save
        session_data = {
            "selected_columns": self.selected_columns,
            "column_renames": self.column_renames,
            "column_descriptions": self.column_descriptions,
            "codebook": self.codebook,
            "examples": self.examples,
            "selected_fields": self.selected_fields,
            "selected_model": self.selected_model,
        }

        # Convert session data to JSON
        data_json = json.dumps(session_data, indent=4)

        # Download button
        st.download_button(
            label="💾 **Save Session Configuration**",
            data=data_json,
            file_name=filename_input,  # Dynamic filename
            mime="application/json",
            key="save_session_button",
        )

    def configure_llm(self) -> None:
        st.header("Step 5: Choose the Model")

        # 🚨 Block Step 5 if Step 4 is incomplete
        if not self.selected_fields:
            st.warning(
                "⚠️ Please specify at least one field to extract in Step 4 before continuing."
            )
            return

        # Step 5.1: Select Provider
        provider_options = ["Select Provider", "OpenAI", "Together", "Azure"]
        selected_provider_display = st.selectbox(
            "Select LLM Provider:", provider_options, key="llm_provider_select"
        )

        if selected_provider_display == "Select Provider":
            st.info("ℹ️ Please select a provider to continue.")
            return

        # Map provider display name to internal config key
        provider_map = {"OpenAI": "openai", "Together": "together", "Azure": "azure"}
        internal_provider = provider_map[selected_provider_display]

        # Step 5.2: Retrieve API Key from config.py
        existing_api_key = config.MODEL_CONFIG[internal_provider].get("api_key")

        # Step 5.3: If not in .env, ask the user for API key
        if existing_api_key:
            st.success(
                f"🔑 API Key successfully loaded from `.env` for {selected_provider_display}!"
            )
            final_api_key = existing_api_key
        else:
            st.sidebar.subheader("API Key Configuration")
            api_key_placeholder = {
                "openai": "sk-...",
                "together": "together-...",
                "azure": "azure-...",
            }.get(internal_provider, "Enter API Key")

            api_key = st.sidebar.text_input(
                f"Enter your {selected_provider_display} API Key",
                type="password",
                placeholder=api_key_placeholder,
                help=f"Provide your {selected_provider_display} API key.",
            )

            # Privacy notice
            st.sidebar.info(
                "🔒 Your API key is used only during this session and is never stored."
            )

            if not api_key:
                st.warning(f"Please provide your {selected_provider_display} API key.")
                st.stop()
            else:
                st.success(f"{selected_provider_display} API Key loaded successfully!")
                final_api_key = api_key

        # Step 5.4: Configure the provider with the API Key
        provider_config = config.MODEL_CONFIG[internal_provider].copy()
        provider_config["api_key"] = final_api_key

        # Step 5.5: Initialize the LLM Client
        self.llm_client = get_llm_client(
            provider=internal_provider, config=provider_config
        )

        # Step 5.6: Select Model
        if selected_provider_display == "OpenAI":
            model_options = ["gpt-4o", "gpt-4o-mini"]
        elif selected_provider_display == "Together":
            model_options = ["gpt-neoxt-chat-20B"]
        else:  # Azure models
            model_options = ["gpt-4o", "gpt-4o-mini"]

        chosen_model = st.selectbox(
            "Select Model:",
            model_options,
            key="llm_model_select",
        )

        self.selected_model = chosen_model
        st.session_state["selected_model"] = chosen_model

    def run_analysis(self) -> None:
        """
        Step 6: Run Analysis
        Uses the selected columns and LLM configuration to perform
        classification or extraction tasks on the processed data.
        Allows users to estimate cost, run debug mode, and save results to CSV.
        """
        st.header("Step 6: Run Analysis")

        # Always rebuild the data_format_description to ensure it matches the latest renamed columns
        data_format_description = build_data_format_description(
            self.column_descriptions
        )
        st.session_state["data_format_description"] = data_format_description

        if self.processed_data is None or self.processed_data.empty:
            st.warning("No processed data. Please go to Step 2.")
            return

        if not self.codebook.strip():
            st.warning("Please provide a codebook in Step 3.")
            return

        if not self.selected_fields:
            st.warning("Please specify the fields to extract in Step 4.")
            return

        if not self.llm_client or not self.selected_model:
            st.warning("Please configure the model in Step 5.")
            return

        # Allow selecting a subset of rows
        st.subheader("Choose how many rows to analyze")
        process_options = ["All rows", "Subset of rows"]
        selected_option = st.radio(
            "Process:", process_options, index=0, key="process_option_radio"
        )

        num_rows = len(self.processed_data)
        if selected_option == "Subset of rows":
            num_rows = st.number_input(
                "Number of rows to process:",
                min_value=1,
                max_value=len(self.processed_data),
                value=min(10, len(self.processed_data)),
                step=1,
                key="num_rows_input",
            )

        # Cost Estimation for the First Entry
        if st.button(
            "Estimate price before analysis (will run on one entry)",
            key="estimate_cost_button",
        ):
            first_entry = self.processed_data.iloc[0]
            entry_text_str = "\n".join(
                [f"{col}: {first_entry[col]}" for col in self.processed_data.columns]
            )

            prompt = construct_prompt(
                data_format_description=data_format_description,
                entry_text=entry_text_str,
                codebook=self.codebook,
                examples=self.examples,
                instructions="You are an assistant that evaluates data entries.",
                selected_fields=self.selected_fields,
                output_format_example={
                    field: "Sample text" for field in self.selected_fields
                },
            )

            try:
                response, usage = self.llm_client.get_response(
                    prompt=prompt,
                    model=self.selected_model,
                    max_tokens=500,
                    temperature=0,
                )
                cost_for_one = openai_api_calculate_cost(usage, self.selected_model)
                total_cost_estimate = cost_for_one * num_rows

                st.info(f"Estimated cost for processing one entry: ${cost_for_one:.4f}")
                st.info(
                    f"Estimated total cost for {num_rows} entries: ${total_cost_estimate:.4f}"
                )

                st.session_state["cost_for_one"] = cost_for_one
                st.session_state["total_cost_estimate"] = total_cost_estimate

            except Exception as e:
                st.error(f"Error estimating cost: {e}")

        # Debug mode checkbox
        debug_mode = st.checkbox(
            "Show constructed prompt for debugging",
            value=False,
            key="debug_mode_checkbox",
        )

        # Start Full Analysis
        if st.button("Run Analysis", key="run_analysis_button"):
            st.info("Processing entries...")
            results: List[Dict[str, Any]] = []
            progress_bar = st.progress(0)

            data_to_process = self.processed_data.head(num_rows)
            total = len(data_to_process)

            for i, (idx, row) in enumerate(data_to_process.iterrows()):
                entry_text_str = "\n".join(
                    [f"{col}: {row[col]}" for col in self.processed_data.columns]
                )

                prompt = construct_prompt(
                    data_format_description=data_format_description,
                    entry_text=entry_text_str,
                    codebook=self.codebook,
                    examples=self.examples,
                    instructions="You are an assistant that evaluates data entries.",
                    selected_fields=self.selected_fields,
                    output_format_example={
                        field: "Your text here" for field in self.selected_fields
                    },
                )

                # If debug mode is on, show the prompt
                if debug_mode:
                    st.write("**Constructed Prompt:**")
                    st.code(prompt)

                try:
                    response, usage = self.llm_client.get_response(
                        prompt=prompt,
                        model=self.selected_model,
                        max_tokens=500,
                        temperature=0,
                        verbose=False,
                    )
                    parsed = parse_llm_response(response, self.selected_fields)
                    results.append({**row.to_dict(), **parsed})
                except Exception as e:
                    st.error(f"Error processing row {idx}: {e}")
                    continue

                progress_bar.progress((i + 1) / total)

            self.results = results
            st.session_state["results"] = results

            st.success("Analysis completed!")
            results_df = pd.DataFrame(results)
            st.session_state["results_df"] = results_df

        # Save Results Section (Moved outside the Run Analysis button block)
        if self.results:
            # Display results as a dataframe preview
            results_df = pd.DataFrame(self.results)
            st.dataframe(results_df)
            # Styled Header (matching the "Save Session" look)
            st.markdown(
                """
                <h4> <b>Save Analysis Results</b></h4>
                <p style='font-size:16px'>
                Download the results of your analysis in Excel format.<br><br>
                """,
                unsafe_allow_html=True,
            )

            # Input for custom filename
            filename_input = st.text_input(
                "**Enter a filename for your results:**",
                value="analysis_results.xlsx",
                key="results_filename_input",
            )

            # Ensure the filename ends with .xlsx
            if not filename_input.endswith(".xlsx"):
                filename_input += ".xlsx"

            # Convert DataFrame to Excel in-memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                results_df.to_excel(writer, index=False, sheet_name="Results")
            data_xlsx = output.getvalue()

            # Download button with dynamic filename
            st.download_button(
                label="💾 **Download Results as Excel**",
                data=data_xlsx,
                file_name=filename_input,  # Dynamic filename
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_results_button",
            )

    def compare_with_external_judgments(self) -> None:
        """
        Step 7: Compare with External Judgments (Optional)
        Allows the user to upload a separate dataset (CSV/XLSX) containing
        external/human judgments to compare against the LLM-generated results.
        """
        st.header("Step 7: Compare with External Judgments (Optional)")
        comparison_file = st.file_uploader(
            "Upload a comparison dataset (CSV or XLSX)",
            type=["csv", "xlsx"],
            key="comparison_file_uploader",
        )

        if comparison_file is not None:
            file_type = "csv" if comparison_file.name.endswith(".csv") else "xlsx"

            # Let the user specify a delimiter for the comparison file if it's CSV
            if file_type == "csv":
                comp_delimiter = st.text_input(
                    "Delimiter for comparison CSV:",
                    value=";",
                    key="comp_delimiter_input",
                )
            else:
                comp_delimiter = None  # not used for XLSX

            try:
                comp_data = load_data(
                    comparison_file,
                    file_type=file_type,
                    delimiter=comp_delimiter if comp_delimiter else ";",
                )
                st.write("Comparison data preview:")
                st.dataframe(comp_data.head())

                # Retrieve results from session_state if needed
                if not self.results:
                    self.results = st.session_state.get("results", [])
                if not self.results:
                    st.error(
                        "No analysis results found. Please run the analysis first."
                    )
                    return

                results_df = pd.DataFrame(self.results)

                # Let user pick the key columns from each DataFrame
                st.subheader("Select key column to merge on (LLM Results)")
                llm_columns = results_df.columns.tolist()
                llm_key_col: str = st.selectbox(
                    "LLM Key Column:", llm_columns, key="llm_key_column_select"
                )

                st.subheader("Select key column to merge on (Comparison Dataset)")
                comp_columns = comp_data.columns.tolist()
                comp_key_col: str = st.selectbox(
                    "Comparison Key Column:", comp_columns, key="comp_key_column_select"
                )

                # Let the user choose which columns from comp_data to keep (besides the key)
                st.subheader(
                    "Select columns from the comparison dataset to include in the merge:"
                )
                possible_comp_cols = [
                    col for col in comp_columns if col != comp_key_col
                ]
                selected_comp_cols = st.multiselect(
                    "Columns to import:",
                    possible_comp_cols,
                    default=possible_comp_cols,
                    key="selected_comp_cols_multiselect",
                )

                if selected_comp_cols:
                    # Convert both sides to string (for consistent merge keys)
                    results_df[llm_key_col] = results_df[llm_key_col].astype(str)
                    comp_data[comp_key_col] = comp_data[comp_key_col].astype(str)

                    # Subset the comparison data to the key + selected columns
                    comp_data_subset = comp_data[[comp_key_col] + selected_comp_cols]

                    # Perform the merge
                    merged = pd.merge(
                        results_df,
                        comp_data_subset,
                        left_on=llm_key_col,
                        right_on=comp_key_col,
                        how="inner",
                    )

                    st.write("Merged Dataframe:")
                    st.dataframe(merged.head())

                    st.subheader("Select columns to compute Cohen's Kappa:")
                    merged_columns = merged.columns.tolist()
                    llm_judgment_col: str = st.selectbox(
                        "LLM Judgment Column:",
                        merged_columns,
                        key="llm_judgment_col_select",
                    )
                    external_judgment_col: str = st.selectbox(
                        "External Judgment Column:",
                        merged_columns,
                        key="external_judgment_col_select",
                    )

                    if st.button("Compute Cohen's Kappa", key="compute_kappa_button"):
                        if (
                            llm_judgment_col not in merged.columns
                            or external_judgment_col not in merged.columns
                        ):
                            st.error("Selected columns not found in merged data.")
                            return

                        # Drop NaNs first
                        judgments_1 = merged[llm_judgment_col].dropna()
                        judgments_2 = merged[external_judgment_col].dropna()

                        # Align the indices after dropping NaNs
                        merged_aligned = pd.concat(
                            [judgments_1, judgments_2], axis=1
                        ).dropna()
                        judgments_1 = merged_aligned[llm_judgment_col].astype(int)
                        judgments_2 = merged_aligned[external_judgment_col].astype(int)

                        if len(judgments_1) == 0 or len(judgments_2) == 0:
                            st.error(
                                "No valid data to compare after converting to int."
                            )
                            return

                        kappa = compute_cohens_kappa(judgments_1, judgments_2)
                        st.write(f"Cohen's Kappa: {kappa:.4f}")

            except Exception as e:
                st.error(f"Error loading comparison file: {e}")


def main() -> None:
    app = QualitativeAnalysisApp()
    app.run()


if __name__ == "__main__":
    main()
