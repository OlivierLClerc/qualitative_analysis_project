# app.py

import streamlit as st
import pandas as pd
from qualitative_analysis.data_processing import (
    load_data, clean_and_normalize, sanitize_dataframe, select_and_rename_columns
)
from qualitative_analysis.prompt_construction import (
    build_data_format_description, construct_prompt
)
from qualitative_analysis.model_interaction import get_llm_client
from qualitative_analysis.response_parsing import parse_llm_response
from qualitative_analysis.utils import save_results_to_csv
import qualitative_analysis.config as config

def main():
    st.title("Qualitative Analysis App")

    # Step 1: Upload Data
    st.header("1. Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'xlsx'

        # Optionally, allow the user to specify the delimiter
        delimiter = st.text_input("Enter the delimiter used in your CSV file:", value=';')

        try:
            data = load_data(uploaded_file, file_type=file_type, delimiter=delimiter)
            st.success("Data loaded successfully!")
            st.write("Preview of the data:")
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        # Step 2: Data Processing Options
        st.header("2. Data Processing Options")

        # Select and Rename Columns
        st.subheader("Select and Rename Columns")
        columns = data.columns.tolist()
        selected_columns = st.multiselect("Select columns to use:", columns)
        column_renames = {}
        for col in selected_columns:
            new_name = st.text_input(f"Rename '{col}' to:", value=col)
            column_renames[col] = new_name

        if st.button("Process Data"):
            if selected_columns:
                # Select and rename columns
                data = select_and_rename_columns(data, selected_columns, column_renames)

                # Clean and normalize text columns
                st.subheader("Clean and Normalize Text Columns")
                text_columns = st.multiselect("Select text columns to clean:", data.columns.tolist(), default=data.columns.tolist())
                for col in text_columns:
                    data[col] = clean_and_normalize(data[col])

                # Sanitize the DataFrame
                data = sanitize_dataframe(data)

                st.success("Data processed successfully!")
                st.write("Processed data:")
                st.dataframe(data.head())
            else:
                st.error("Please select at least one column.")

        # Proceed only if data has been processed
        if 'data' in locals():
            # Step 3: Provide Codebook and Examples
            st.header("3. Provide Codebook and Examples")

            codebook = st.text_area("Enter your codebook (instructions for the LLM):")
            examples = st.text_area("Enter examples (optional):")

            # Step 4: Configure the LLM
            st.header("4. Configure the Language Model")

            provider_options = ['OpenAI', 'Together']
            selected_display_name = st.selectbox('Select the provider:', options=provider_options)

            # Map the displayed provider name to the internal provider name
            name_mapping = {
                'OpenAI': 'azure',
                'Together': 'together'
            }

            internal_provider = name_mapping[selected_display_name]

            model_options = []
            if selected_display_name == 'OpenAI':
                model_options = ['gpt-4o', 'gpt-4o-mini']
            # Initialize the LLM client
            llm_client = get_llm_client(provider=internal_provider, config=config.MODEL_CONFIG[internal_provider])
            selected_model = st.selectbox('Select the model:', options=model_options)

            # Step 5: Run Analysis
            st.header("5. Run Analysis")

            if st.button("Run Analysis"):
                if codebook.strip() == "":
                    st.error("Please provide a codebook.")
                else:
                    st.info("Processing entries...")
                    results = []
                    # Optional: Add a progress bar
                    progress_bar = st.progress(0)
                    total_entries = len(data)
                    for idx, (index, row) in enumerate(data.iterrows()):
                        # Build the prompt
                        column_descriptions = {col: f"Column {col}" for col in data.columns}
                        data_format_description = build_data_format_description(column_descriptions)
                        entry_text = row.to_dict()
                        entry_text_str = '\n'.join(f"{k}: {v}" for k, v in entry_text.items())

                        prompt = construct_prompt(
                            data_format_description=data_format_description,
                            entry_text=entry_text_str,
                            codebook=codebook,
                            examples=examples,
                            instructions="You are an assistant that evaluates data entries.",
                            selected_fields=['Evaluation', 'Comments'],
                            output_format_example={'Evaluation': 'Positive', 'Comments': 'Well-written entry.'}
                        )

                        # Get the response from the LLM
                        try:
                            response = llm_client.get_response(
                                prompt=prompt,
                                model=selected_model,
                                max_tokens=500,
                                temperature=0,
                                verbose=False
                            )
                            # Parse the response
                            parsed_response = parse_llm_response(response, selected_fields=['Evaluation', 'Comments'])
                            results.append({**entry_text, **parsed_response})
                        except Exception as e:
                            st.error(f"Error processing entry {index}: {e}")
                            continue
                        # Update progress bar
                        progress_bar.progress((idx + 1) / total_entries)

                    # Display results
                    st.success("Analysis completed!")
                    results_df = pd.DataFrame(results)
                    st.write("Results:")
                    st.dataframe(results_df)

                    # Step 6: Save Results
                    st.header("6. Save Results")
                    if st.button("Save Results to CSV"):
                        save_path = st.text_input("Enter the filename (e.g., results.csv):", value="results.csv")
                        if save_path:
                            save_results_to_csv(
                                coding=results,
                                save_path=save_path,
                                fieldnames=list(results_df.columns),
                                verbatims=None  # Assuming verbatims are included in the results
                            )
                            st.success(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()