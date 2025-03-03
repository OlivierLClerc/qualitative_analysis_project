"""
Module for handling analysis functionality in the Streamlit app.
"""

import streamlit as st
import pandas as pd
import io
from typing import Any, Optional

from qualitative_analysis import (
    build_data_format_description,
    construct_prompt,
    parse_llm_response,
    openai_api_calculate_cost,
)


def run_analysis(app_instance: Any) -> Optional[pd.DataFrame]:
    """
    Step 6: Run Analysis
    Uses the selected columns and LLM configuration to perform
    classification or extraction tasks on the processed data.
    Allows users to estimate cost, run debug mode, and save results to CSV.

    Args:
        app_instance: The QualitativeAnalysisApp instance

    Returns:
        The results DataFrame or None if analysis was not run
    """
    st.header("Step 6: Run Analysis")

    # ------------------------------------------------------------------
    # Validate that we have the necessary data, codebook, fields, etc.
    # ------------------------------------------------------------------
    if app_instance.processed_data is None or app_instance.processed_data.empty:
        st.warning("No processed data. Please go to Step 2.")
        return None

    if not app_instance.codebook.strip():
        st.warning("Please provide a codebook in Step 3.")
        return None

    if not app_instance.selected_fields:
        st.warning("Please specify the fields to extract in Step 4.")
        return None

    if not app_instance.llm_client or not app_instance.selected_model:
        st.warning("Please configure the model in Step 5.")
        return None

    # ------------------------------------------------------------------
    # Build the data format description for the prompt
    # ------------------------------------------------------------------
    data_format_description = build_data_format_description(
        app_instance.column_descriptions
    )
    st.session_state["data_format_description"] = data_format_description

    # ------------------------------------------------------------------
    # Let user pick how many rows to process
    # ------------------------------------------------------------------
    st.subheader("Choose how many rows to analyze")
    process_options = ["All rows", "Subset of rows"]
    selected_option = st.radio(
        "Process:", process_options, index=0, key="process_option_radio"
    )

    num_rows = len(app_instance.processed_data)
    if selected_option == "Subset of rows":
        num_rows = st.number_input(
            "Number of rows to process:",
            min_value=1,
            max_value=len(app_instance.processed_data),
            value=min(10, len(app_instance.processed_data)),
            step=1,
            key="num_rows_input",
        )

    # ------------------------------------------------------------------
    # Cost Estimation button (optional, for one entry)
    # ------------------------------------------------------------------
    if st.button("Estimate price before analysis (will run on one entry)"):
        # Just process the first row for cost estimation
        first_entry = app_instance.processed_data.iloc[0]
        # Build the text from selected/renamed columns
        analysis_cols_renamed = list(app_instance.column_renames.values())
        entry_text_str = "\n".join(
            [
                f"{col}: {first_entry[col]}"
                for col in analysis_cols_renamed
                if col in app_instance.processed_data.columns
            ]
        )

        # Construct the prompt
        prompt = construct_prompt(
            data_format_description=data_format_description,
            entry_text=entry_text_str,
            codebook=app_instance.codebook,
            examples=app_instance.examples,
            instructions="You are an assistant that evaluates data entries.",
            selected_fields=app_instance.selected_fields,
            output_format_example={
                field: "Sample text" for field in app_instance.selected_fields
            },
        )

        try:
            response, usage = app_instance.llm_client.get_response(
                prompt=prompt,
                model=app_instance.selected_model,
                max_tokens=500,
                temperature=0,
            )
            cost_for_one = openai_api_calculate_cost(usage, app_instance.selected_model)
            total_cost_estimate = cost_for_one * num_rows

            st.info(f"Estimated cost for processing one entry: ${cost_for_one:.4f}")
            st.info(
                f"Estimated total cost for {num_rows} entries: ${total_cost_estimate:.4f}"
            )

            st.session_state["cost_for_one"] = cost_for_one
            st.session_state["total_cost_estimate"] = total_cost_estimate

        except Exception as e:
            st.error(f"Error estimating cost: {e}")

    # ------------------------------------------------------------------
    # Debug mode (to show the constructed prompt for each row)
    # ------------------------------------------------------------------
    debug_mode = st.checkbox(
        "Show constructed prompt for debugging",
        value=False,
        key="debug_mode_checkbox",
    )

    # ------------------------------------------------------------------
    # Run Full Analysis
    # ------------------------------------------------------------------
    if st.button("Run Analysis", key="run_analysis_button"):
        st.info("Processing entries...")

        results = []
        progress_bar = st.progress(0)
        data_to_process = app_instance.processed_data.head(num_rows)
        total = len(data_to_process)

        # We'll fetch only the renamed columns that the user is analyzing
        analysis_cols_renamed = list(
            set(app_instance.column_renames.values()).intersection(
                data_to_process.columns
            )
        )

        for i, (idx, row) in enumerate(data_to_process.iterrows()):
            try:
                # Build the text from the selected (renamed) columns
                entry_text_str = "\n".join(
                    [f"{col}: {row[col]}" for col in analysis_cols_renamed]
                )

                # Construct the prompt
                prompt = construct_prompt(
                    data_format_description=data_format_description,
                    entry_text=entry_text_str,
                    codebook=app_instance.codebook,
                    examples=app_instance.examples,
                    instructions="You are an assistant that evaluates data entries.",
                    selected_fields=app_instance.selected_fields,
                    output_format_example={
                        field: "Your text here"
                        for field in app_instance.selected_fields
                    },
                )

                if debug_mode:
                    st.write("**Constructed Prompt:**")
                    st.code(prompt)

                # Call the LLM
                response, usage = app_instance.llm_client.get_response(
                    prompt=prompt,
                    model=app_instance.selected_model,
                    max_tokens=500,
                    temperature=0,
                    verbose=False,
                )

                # Parse the LLM response into a dictionary
                parsed = parse_llm_response(response, app_instance.selected_fields)

                # ------------------------------------------------------------------
                # Optional: partial numeric extraction from label column
                # (If label_type is 'Integer' or 'Float', tries to extract a number)
                # ------------------------------------------------------------------
                label_column = st.session_state.get("label_column")
                label_type = st.session_state.get("label_type")

                if label_column and label_column in parsed:

                    def extract_numeric_value(value, target_type):
                        import re

                        if isinstance(value, (int, float)):
                            return value

                        str_value = str(value)
                        # Look for integer first
                        int_match = re.search(r"\b(\d+)\b", str_value)
                        if int_match:
                            try:
                                return (
                                    int(int_match.group(1))
                                    if target_type == "Integer"
                                    else float(int_match.group(1))
                                )
                            except ValueError:
                                pass

                        # Look for decimal
                        float_match = re.search(r"\b(\d+\.\d+)\b", str_value)
                        if float_match:
                            try:
                                return (
                                    int(float_match.group(1).split(".")[0])
                                    if target_type == "Integer"
                                    else float(float_match.group(1))
                                )
                            except ValueError:
                                pass

                        # If no numeric value found, return None
                        return None

                    # Attempt best-effort numeric conversion
                    if label_type == "Integer":
                        converted_value = extract_numeric_value(
                            parsed[label_column], "Integer"
                        )
                        if converted_value is not None:
                            parsed[label_column] = converted_value
                        else:
                            st.warning(
                                f"Could not extract an integer from '{parsed[label_column]}' at row {idx}. Keeping as text."
                            )
                    elif label_type == "Float":
                        converted_value = extract_numeric_value(
                            parsed[label_column], "Float"
                        )
                        if converted_value is not None:
                            parsed[label_column] = converted_value
                        else:
                            st.warning(
                                f"Could not extract a float from '{parsed[label_column]}' at row {idx}. Keeping as text."
                            )
                    else:
                        # If "Text", just ensure it's string
                        parsed[label_column] = str(parsed[label_column])

                # ------------------------------------------------------------------
                # Merge in annotation columns that might not be in row.to_dict()
                # If your app_instance.processed_data does NOT have them, fetch from app_instance.data:
                # ------------------------------------------------------------------
                annotation_dict = {}
                if app_instance.data is None:
                    # Possibly raise an error or return
                    raise ValueError(
                        "app_instance.data is None, cannot merge annotation columns."
                    )

                for ann_col in app_instance.annotation_columns:
                    if ann_col in app_instance.data.columns:
                        annotation_dict[ann_col] = app_instance.data.loc[idx, ann_col]

                # Combine everything: original row data, LLM parsed fields, annotation columns
                combined = {**row.to_dict(), **parsed, **annotation_dict}
                results.append(combined)

            except Exception as e:
                st.error(f"Error processing row {idx}: {e}")
                continue

            progress_bar.progress((i + 1) / total)

        # ------------------------------------------------------------------
        # Build results_df from all processed rows
        # ------------------------------------------------------------------
        results_df = pd.DataFrame(results)
        app_instance.results = results
        st.session_state["results"] = results

        st.success("Analysis completed!")
        # ------------------------------------------------------------------
        # Final Type Conversions for annotation cols and label col
        # ------------------------------------------------------------------
        label_type = st.session_state.get("label_type", None)
        label_column = st.session_state.get("label_column", None)

        if label_type:
            for ann_col in app_instance.annotation_columns:
                if ann_col in results_df.columns:
                    try:
                        if label_type == "Integer":
                            results_df[ann_col] = results_df[ann_col].astype(int)
                        elif label_type == "Float":
                            results_df[ann_col] = results_df[ann_col].astype(float)
                        else:  # treat as string
                            results_df[ann_col] = results_df[ann_col].astype(str)
                    except (ValueError, TypeError):
                        st.warning(
                            f"Could not fully convert annotation column '{ann_col}' to {label_type}."
                        )

            if label_column and label_column in results_df.columns:
                try:
                    if label_type == "Integer":
                        results_df[label_column] = results_df[label_column].astype(int)
                    elif label_type == "Float":
                        results_df[label_column] = results_df[label_column].astype(
                            float
                        )
                    else:  # text
                        results_df[label_column] = results_df[label_column].astype(str)
                except (ValueError, TypeError):
                    st.warning(
                        f"Could not convert LLM label column '{label_column}' to {label_type}."
                    )

        # ------------------------------------------------------------------
        # Store final DataFrame in session and display a preview
        # ------------------------------------------------------------------
        st.session_state["results_df"] = results_df
        st.dataframe(results_df.head())

        # Optional: Provide a download button for results
        filename_input = st.text_input(
            "**Enter a filename for your results:**",
            value="analysis_results.xlsx",
            key="results_filename_input",
        )
        if not filename_input.endswith(".xlsx"):
            filename_input += ".xlsx"

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            results_df.to_excel(writer, index=False, sheet_name="Results")
        data_xlsx = output.getvalue()

        st.download_button(
            label="💾 **Download Results as Excel**",
            data=data_xlsx,
            file_name=filename_input,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_results_button",
        )

        return results_df

    return None
