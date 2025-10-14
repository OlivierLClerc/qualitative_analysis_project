"""
Module for handling content generation using LLM.
"""

import streamlit as st
import pandas as pd
import json
from typing import Any, Optional, Dict, List

from qualitative_analysis import (
    parse_llm_response,
    openai_api_calculate_cost,
)


def build_generation_prompt(
    blueprints: Dict[str, str],
    generation_prompt: str,
    column_descriptions: Dict[str, str],
    columns: List[str],
) -> str:
    """
    Build a prompt for content generation based on blueprints and configuration.

    Args:
        blueprints: Dictionary of blueprint examples (column_name -> blueprint_text)
        generation_prompt: User's prompt describing desired variations
        column_descriptions: Dictionary mapping column names to descriptions
        columns: List of column names for output

    Returns:
        Formatted prompt for LLM
    """
    # Format blueprints section (without labels, just the text)
    blueprints_section = ""
    for blueprint_text in blueprints.values():
        blueprints_section += f"{blueprint_text}\n\n"

    # Build output format example using the actual column descriptions
    output_example = {}
    for col in columns:
        desc = column_descriptions.get(col, "")
        # Use the description as the example value if provided, otherwise use a generic placeholder
        output_example[col] = desc if desc else f"Your generated {col}"

    prompt = f"""**Generation Instructions:**
{generation_prompt}

{blueprints_section}

**IMPORTANT:** Your response must be valid JSON only, containing a single object with the specified columns. Do not include any explanations, greetings, or additional text.

**Example output format:**
{json.dumps(output_example, ensure_ascii=False, indent=2)}"""

    return prompt


def run_generation(app_instance: Any) -> Optional[pd.DataFrame]:
    """
    Step 4: Content Generation
    Execute the content generation process using the configured LLM.

    Args:
        app_instance: The QualitativeAnalysisApp instance

    Returns:
        DataFrame containing generated content or None if generation failed
    """
    st.markdown("### Step 4: Content Generation", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 4", expanded=True):

        # Check prerequisites
        if not hasattr(app_instance, "blueprints") or not app_instance.blueprints:
            st.warning("Please provide blueprint examples in Step 1 first.")
            return None

        if (
            not hasattr(app_instance, "generation_config")
            or not app_instance.generation_config
        ):
            st.warning("Please configure generation settings in Step 2 first.")
            return None

        if not app_instance.llm_client or not app_instance.selected_model:
            st.warning("Please configure the LLM in the previous steps.")
            return None

        config = app_instance.generation_config
        columns = list(config["columns"].keys())

        # Number of items to generate
        st.subheader("Generation Settings")
        num_items = st.number_input(
            "**Number of items to generate:**",
            min_value=1,
            max_value=100,
            value=st.session_state.get("num_items_to_generate", 5),
            step=1,
            key="num_items_input",
            help="Each item requires a separate API call to the LLM",
        )
        st.session_state["num_items_to_generate"] = num_items

        # Update the config with the number of items
        config["num_items"] = num_items

        # Cost estimation
        st.subheader("Cost Estimation")
        st.markdown(
            """
            **Note:** Each item requires a separate API call. The cost estimation below is based on generating one item,
            then multiplied by the total number of items requested.
            """,
            unsafe_allow_html=True,
        )
        if st.button("Estimate generation cost", key="estimate_generation_cost"):
            try:
                # Generate one sample for cost estimation
                sample_prompt = build_generation_prompt(
                    app_instance.blueprints,
                    config["generation_prompt"],
                    config["column_descriptions"],
                    columns,
                )

                response, usage = app_instance.llm_client.get_response(
                    prompt=sample_prompt,
                    model=app_instance.selected_model,
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                    verbose=False,
                )

                cost_for_one = openai_api_calculate_cost(
                    usage, app_instance.selected_model
                )
                total_cost_estimate = cost_for_one * config["num_items"]

                st.info(f"💰 Cost per item (1 API call): ${cost_for_one:.4f}")
                st.info(
                    f"💰 Total cost ({config['num_items']} API calls): ${total_cost_estimate:.4f}"
                )

                st.session_state["generation_cost_estimate"] = total_cost_estimate

            except Exception as e:
                st.error(f"Error estimating cost: {e}")

        # Debug mode option
        debug_mode = st.checkbox(
            "Show generation prompts for debugging",
            value=False,
            key="generation_debug_mode",
        )

        # Generate content button
        st.subheader("Generate Content")
        if st.button("🚀 Start Content Generation", key="start_generation_button"):

            st.info(f"Generating {config['num_items']} items...")

            # Progress tracking
            results = []
            progress_bar = st.progress(0)

            # Status containers
            status_container = st.empty()
            error_container = st.empty()

            successful_generations = 0
            failed_generations = 0

            for i in range(config["num_items"]):
                try:
                    status_container.text(
                        f"Generating item {i + 1} of {config['num_items']}..."
                    )

                    # Build prompt for this generation
                    prompt = build_generation_prompt(
                        app_instance.blueprints,
                        config["generation_prompt"],
                        config["column_descriptions"],
                        columns,
                    )

                    if debug_mode:
                        st.write(f"**Generation Prompt (item {i + 1}):**")
                        st.code(prompt)

                    # Call LLM
                    response, usage = app_instance.llm_client.get_response(
                        prompt=prompt,
                        model=app_instance.selected_model,
                        max_tokens=config["max_tokens"],
                        temperature=config["temperature"],
                        verbose=False,
                    )

                    # Parse response
                    try:
                        # Try to parse as JSON first
                        parsed_data = json.loads(response.strip())

                        # Ensure all expected columns are present
                        result_row = {}
                        for col in columns:
                            result_row[col] = parsed_data.get(col, "")

                        # Add metadata
                        result_row["generation_id"] = i + 1

                        # Add blueprint columns
                        for (
                            blueprint_col,
                            blueprint_text,
                        ) in app_instance.blueprints.items():
                            result_row[blueprint_col] = blueprint_text

                        results.append(result_row)
                        successful_generations += 1

                    except json.JSONDecodeError:
                        # Fallback: try to extract using existing parse function
                        try:
                            parsed_data = parse_llm_response(response, columns)
                            result_row = {}
                            for col in columns:
                                result_row[col] = parsed_data.get(col, "")
                            result_row["generation_id"] = i + 1

                            # Add blueprint columns in fallback path too
                            for (
                                blueprint_col,
                                blueprint_text,
                            ) in app_instance.blueprints.items():
                                result_row[blueprint_col] = blueprint_text

                            results.append(result_row)
                            successful_generations += 1
                        except Exception as parse_error:
                            st.warning(
                                f"Failed to parse response for item {i + 1}: {parse_error}"
                            )
                            failed_generations += 1
                            continue

                except Exception as e:
                    error_container.error(f"Error generating item {i + 1}: {e}")
                    failed_generations += 1
                    continue

                # Update progress
                progress_bar.progress((i + 1) / config["num_items"])

            # Clear status
            status_container.empty()

            # Show results
            if results:
                generated_df = pd.DataFrame(results)

                st.success("✅ Generation completed!")
                st.info(f"Successfully generated: {successful_generations} items")
                if failed_generations > 0:
                    st.warning(f"Failed generations: {failed_generations} items")

                # Store results
                app_instance.generated_content = generated_df
                st.session_state["generated_content"] = generated_df
                st.session_state["generation_completed"] = True

                # Display results
                st.subheader("Generated Content Preview")
                st.dataframe(generated_df.head(10))

                if len(generated_df) > 10:
                    st.info(
                        f"Showing first 10 rows. Total generated: {len(generated_df)} items."
                    )

                # Download option
                st.subheader("Download Generated Content")
                filename = st.text_input(
                    "Filename for generated content:",
                    value="generated_content.xlsx",
                    key="generated_content_filename",
                )

                if not filename.endswith((".xlsx", ".csv")):
                    filename += ".xlsx"

                # Export to Excel
                import io

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    generated_df.to_excel(
                        writer, index=False, sheet_name="Generated Content"
                    )
                data_xlsx = output.getvalue()

                st.download_button(
                    label="💾 Download Generated Content",
                    data=data_xlsx,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_generated_content_button",
                )

                return generated_df

            else:
                st.error(
                    "No content was successfully generated. Please check your configuration and try again."
                )
                return None

        # Display existing results if available
        if (
            st.session_state.get("generation_completed", False)
            and st.session_state.get("generated_content") is not None
        ):
            generated_df = st.session_state["generated_content"]

            st.subheader("Previously Generated Content")
            st.dataframe(generated_df.head(10))

            if len(generated_df) > 10:
                st.info(
                    f"Showing first 10 rows. Total generated: {len(generated_df)} items."
                )

            return generated_df

    return None
