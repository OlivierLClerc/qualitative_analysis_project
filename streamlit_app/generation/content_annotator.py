"""
Module for handling content annotation in the generation workflow.
"""

import streamlit as st
import pandas as pd
import json
import io
from typing import Any, Optional, Dict, List

from qualitative_analysis import (
    parse_llm_response,
    openai_api_calculate_cost,
)


def build_annotation_prompt(
    content_row: Dict[str, Any],
    blueprints: Dict[str, str],
    codebook: str,
    examples: str,
    annotation_fields: List[str],
) -> str:
    """
    Build a prompt for annotating generated content using codebook format from annotation mode.

    Args:
        content_row: Dictionary representing a row of generated content
        blueprints: Dictionary of blueprint examples for reference
        codebook: The annotation codebook from Step 6
        examples: Optional examples from Step 6
        annotation_fields: List of field names to extract from Step 7

    Returns:
        Formatted prompt for LLM annotation
    """
    # Format the content to be annotated (excluding blueprint columns and metadata)
    content_section = "Here is the generated content to annotate:\n\n"
    blueprint_col_names = list(blueprints.keys())
    for key, value in content_row.items():
        if (
            key != "generation_id" and key not in blueprint_col_names
        ):  # Skip metadata and blueprint columns
            content_section += f"**{key}**: {value}\n"

    # Format blueprints section for reference (from the blueprint columns in the row)
    blueprints_section = (
        "For reference, here are the original correct blueprint examples:\n\n"
    )
    for blueprint_col in blueprint_col_names:
        if blueprint_col in content_row:
            blueprints_section += f"{content_row[blueprint_col]}\n\n"

    # Build output format example
    output_example = {
        field: f"Your annotation for {field}" for field in annotation_fields
    }

    prompt = f"""{codebook}

{examples}

{content_section}

{blueprints_section}

**Instructions:**
- Evaluate the generated content according to the codebook and examples.
- Your response should include the following fields: {', '.join(annotation_fields)}.
- **Your response must be in JSON format only. Do not include any explanations, greetings, or additional text.**

**Example response format:**
{json.dumps(output_example, indent=2)}"""

    return prompt


def annotate_generated_content(app_instance: Any) -> Optional[pd.DataFrame]:
    """
    Step 9: Content Annotation
    Execute the annotation process on the generated content using the codebook and fields defined in Steps 6 and 7.

    Args:
        app_instance: The QualitativeAnalysisApp instance

    Returns:
        DataFrame containing generated content with annotations or None if annotation failed
    """
    st.markdown("### Step 9: Content Annotation", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 9", expanded=True):

        # Check prerequisites
        if (
            not hasattr(app_instance, "generated_content")
            or app_instance.generated_content is None
        ):
            st.warning("Please generate content in Step 4 first.")
            return None

        if (
            not hasattr(app_instance, "selected_columns")
            or not app_instance.selected_columns
        ):
            st.warning("Please select columns for annotation in Step 5 first.")
            return None

        if not app_instance.codebook or not app_instance.codebook.strip():
            st.warning("Please provide a codebook in Step 6 first.")
            return None

        if not app_instance.selected_fields:
            st.warning("Please specify fields to extract in Step 7 first.")
            return None

        if not app_instance.llm_client or not app_instance.selected_model:
            st.warning("Please configure the LLM in the previous steps.")
            return None

        generated_df = app_instance.generated_content.copy()
        annotation_fields = app_instance.selected_fields

        # Annotation parameters (using defaults suitable for consistent annotation)
        annotation_temperature = 0.0
        annotation_max_tokens = 500

        st.markdown(
            """
            **Annotation Setup:**
            Using the codebook from Step 5 and fields from Step 6 to annotate generated content.
            Temperature is set to 0.0 for consistent annotations.
            """,
            unsafe_allow_html=True,
        )

        # Cost estimation
        st.subheader("Cost Estimation")
        if st.button("Estimate annotation cost", key="estimate_annotation_cost"):
            try:
                # Annotate one sample for cost estimation
                sample_row = generated_df.iloc[0].to_dict()
                sample_prompt = build_annotation_prompt(
                    sample_row,
                    app_instance.blueprints,
                    app_instance.codebook,
                    app_instance.examples,
                    annotation_fields,
                )

                response, usage = app_instance.llm_client.get_response(
                    prompt=sample_prompt,
                    model=app_instance.selected_model,
                    max_tokens=annotation_max_tokens,
                    temperature=annotation_temperature,
                    verbose=False,
                )

                cost_for_one = openai_api_calculate_cost(
                    usage, app_instance.selected_model
                )
                total_cost_estimate = cost_for_one * len(generated_df)

                st.info(f"Estimated cost for annotating one item: ${cost_for_one:.4f}")
                st.info(
                    f"Estimated total cost for {len(generated_df)} items: ${total_cost_estimate:.4f}"
                )

                st.session_state["annotation_cost_estimate"] = total_cost_estimate

            except Exception as e:
                st.error(f"Error estimating cost: {e}")

        # Debug mode option
        debug_mode = st.checkbox(
            "Show annotation prompts for debugging",
            value=False,
            key="annotation_debug_mode",
        )

        # Annotate content button
        st.subheader("Annotate Generated Content")
        if st.button("🏷️ Start Content Annotation", key="start_annotation_button"):

            st.info(f"Annotating {len(generated_df)} items...")

            # Progress tracking
            annotated_results = []
            progress_bar = st.progress(0)

            # Status containers
            status_container = st.empty()
            error_container = st.empty()

            successful_annotations = 0
            failed_annotations = 0

            for i, (idx, row) in enumerate(generated_df.iterrows()):
                try:
                    status_container.text(
                        f"Annotating item {i + 1} of {len(generated_df)}..."
                    )

                    # Build prompt for this annotation
                    row_dict = row.to_dict()
                    prompt = build_annotation_prompt(
                        row_dict,
                        app_instance.blueprints,
                        app_instance.codebook,
                        app_instance.examples,
                        annotation_fields,
                    )

                    if debug_mode:
                        st.write(f"**Annotation Prompt (item {i + 1}):**")
                        st.code(prompt)

                    # Call LLM
                    response, usage = app_instance.llm_client.get_response(
                        prompt=prompt,
                        model=app_instance.selected_model,
                        max_tokens=annotation_max_tokens,
                        temperature=annotation_temperature,
                        verbose=False,
                    )

                    # Parse response
                    try:
                        # Try to parse as JSON first
                        parsed_annotation = json.loads(response.strip())

                        # Ensure all expected annotation fields are present
                        annotation_data = {}
                        for field in annotation_fields:
                            annotation_data[field] = parsed_annotation.get(field, "")

                        # Combine original row with annotation data
                        combined_row = {**row_dict, **annotation_data}
                        annotated_results.append(combined_row)
                        successful_annotations += 1

                    except json.JSONDecodeError:
                        # Fallback: try to extract using existing parse function
                        try:
                            parsed_annotation = parse_llm_response(
                                response, annotation_fields
                            )
                            annotation_data = {}
                            for field in annotation_fields:
                                annotation_data[field] = parsed_annotation.get(
                                    field, ""
                                )

                            combined_row = {**row_dict, **annotation_data}
                            annotated_results.append(combined_row)
                            successful_annotations += 1
                        except Exception as parse_error:
                            st.warning(
                                f"Failed to parse annotation for item {i + 1}: {parse_error}"
                            )
                            # Keep original row without annotations
                            failed_row = row_dict.copy()
                            for field in annotation_fields:
                                failed_row[field] = ""
                            annotated_results.append(failed_row)
                            failed_annotations += 1
                            continue

                except Exception as e:
                    error_container.error(f"Error annotating item {i + 1}: {e}")
                    # Keep original row without annotations
                    failed_row = row.to_dict()
                    for field in annotation_fields:
                        failed_row[field] = ""
                    annotated_results.append(failed_row)
                    failed_annotations += 1
                    continue

                # Update progress
                progress_bar.progress((i + 1) / len(generated_df))

            # Clear status
            status_container.empty()

            # Show results
            if annotated_results:
                annotated_df = pd.DataFrame(annotated_results)

                st.success("✅ Annotation completed!")
                st.info(f"Successfully annotated: {successful_annotations} items")
                if failed_annotations > 0:
                    st.warning(f"Failed annotations: {failed_annotations} items")

                # Store results
                app_instance.annotated_content = annotated_df
                st.session_state["annotated_content"] = annotated_df
                st.session_state["annotation_completed"] = True

                # Display results
                st.subheader("Annotated Content Preview")
                st.dataframe(annotated_df.head(10))

                if len(annotated_df) > 10:
                    st.info(
                        f"Showing first 10 rows. Total annotated: {len(annotated_df)} items."
                    )

                # Show column breakdown
                original_columns = [col for col in generated_df.columns]
                new_annotation_fields = annotation_fields

                st.markdown("### **Dataset Structure**")
                st.write(
                    f"**Original columns ({len(original_columns)}):** {', '.join(original_columns)}"
                )
                st.write(
                    f"**New annotation fields ({len(new_annotation_fields)}):** {', '.join(new_annotation_fields)}"
                )
                st.write(f"**Total columns:** {len(annotated_df.columns)}")

                # Download option
                st.subheader("Download Annotated Content")
                filename = st.text_input(
                    "Filename for annotated content:",
                    value="generated_and_annotated_content.xlsx",
                    key="annotated_content_filename",
                )

                if not filename.endswith((".xlsx", ".csv")):
                    filename += ".xlsx"

                # Export to Excel with multiple sheets
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    # Full annotated dataset
                    annotated_df.to_excel(
                        writer, index=False, sheet_name="Annotated Content"
                    )

                    # Original generated content only
                    generated_df.to_excel(
                        writer, index=False, sheet_name="Original Generated"
                    )

                    # Annotations only
                    if len(new_annotation_fields) > 0:
                        annotation_only_df = annotated_df[
                            ["generation_id"] + new_annotation_fields
                        ]
                        annotation_only_df.to_excel(
                            writer, index=False, sheet_name="Annotations Only"
                        )

                data_xlsx = output.getvalue()

                st.download_button(
                    label="💾 Download Annotated Content",
                    data=data_xlsx,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_annotated_content_button",
                )

                return annotated_df

            else:
                st.error(
                    "No content was successfully annotated. Please check your configuration and try again."
                )
                return None

        # Display existing results if available
        if (
            st.session_state.get("annotation_completed", False)
            and st.session_state.get("annotated_content") is not None
        ):
            annotated_df = st.session_state["annotated_content"]

            st.subheader("Previously Annotated Content")
            st.dataframe(annotated_df.head(10))

            if len(annotated_df) > 10:
                st.info(
                    f"Showing first 10 rows. Total annotated: {len(annotated_df)} items."
                )

            return annotated_df

    return None
