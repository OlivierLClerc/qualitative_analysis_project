"""
Module for handling generation configuration in the generation workflow.
"""

import streamlit as st
from typing import Any, Dict, Optional


def configure_generation(app_instance: Any) -> Optional[Dict[str, Any]]:
    """
    Step 2: Generation Configuration
    Allows users to configure what content to generate and how.

    Args:
        app_instance: The QualitativeAnalysisApp instance

    Returns:
        Dictionary containing generation configuration or None if incomplete
    """
    st.markdown("### Step 2: Generation Configuration", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 2", expanded=True):

        # Check if blueprints are available
        if not hasattr(app_instance, "blueprints") or not app_instance.blueprints:
            st.warning("Please provide blueprint examples in Step 1 first.")
            return None

        # Check for gameplay mode and pre-populate
        use_gameplay = st.session_state.get("generation_use_gameplay", False)
        gameplay_data = st.session_state.get("generation_gameplay_data", None)

        if use_gameplay and gameplay_data:
            selected_gameplay = st.session_state.get(
                "generation_selected_gameplay", "Unknown"
            )
            st.info(
                f"**Gameplay Mode Active** - Values pre-populated from template for gameplay: {selected_gameplay}"
            )
            st.markdown("*You can review and modify all pre-filled values below.*")

            # Pre-populate generation prompt
            if (
                "generation_prompt" not in st.session_state
                or not st.session_state["generation_prompt"]
            ):
                generation_prompt_from_config = gameplay_data.get(
                    "generation_prompt",
                    gameplay_data.get("generation", {}).get("prompt", ""),
                )
                if generation_prompt_from_config:
                    st.session_state["generation_prompt"] = (
                        generation_prompt_from_config
                    )

            # Pre-populate output columns
            output_columns_from_config = gameplay_data.get(
                "output_columns",
                gameplay_data.get("generation", {}).get("output_columns", []),
            )
            if output_columns_from_config and (
                "num_generation_columns" not in st.session_state
                or not st.session_state.get("num_generation_columns")
            ):
                st.session_state["num_generation_columns"] = len(
                    output_columns_from_config
                )

                # Pre-populate column names and descriptions
                for i, col_config in enumerate(output_columns_from_config):
                    if isinstance(col_config, dict):
                        st.session_state[f"gen_col_name_{i}"] = col_config.get(
                            "name", f"Column_{i+1}"
                        )
                        st.session_state[f"gen_col_desc_{i}"] = col_config.get(
                            "description", ""
                        )
                    elif isinstance(col_config, str):
                        # If just column names without descriptions
                        st.session_state[f"gen_col_name_{i}"] = col_config
                        st.session_state[f"gen_col_desc_{i}"] = ""

        st.markdown(
            """
            ### **Generation Setup**
            Configure how the LLM should generate new content based on your blueprints.
            
            **How it works:**
            - The **Generation Prompt** describes the type of variation you want (e.g., different values, difficulty levels)
            - The **Number of items** determines how many separate API calls will be made
            - Each API call generates **ONE item** using your prompt and blueprints
            """,
            unsafe_allow_html=True,
        )

        # Generation prompt
        default_generation_prompt = st.session_state.get("generation_prompt", "")
        generation_prompt = st.text_area(
            "**Generation Prompt (applied to each item):**",
            value=default_generation_prompt,
            height=200,
            key="generation_prompt_textarea",
            help="Describe the type of variation for each generated item (e.g., 'Generate a similar exercise with different numerical values and random difficulty level'). This prompt will be used for EACH item generated.",
        )

        if not generation_prompt.strip():
            st.info("Please provide a generation prompt to proceed.")
            return None

        st.markdown("### **Output Columns Configuration**")
        st.markdown(
            """
            Define the structure of your generated content by specifying column names and descriptions.
            Each generated item will be a row with these columns.
            """,
            unsafe_allow_html=True,
        )

        # Column configuration
        num_columns = st.number_input(
            "Number of output columns:",
            min_value=1,
            max_value=20,
            value=st.session_state.get("num_generation_columns", 2),
            step=1,
            key="num_generation_columns_input",
        )

        st.session_state["num_generation_columns"] = num_columns

        # Convert to int for range usage
        num_columns_int = int(num_columns)

        generation_columns: Dict[str, int] = {}
        column_descriptions: Dict[str, str] = {}

        for i in range(num_columns_int):
            col1, col2 = st.columns([1, 2])

            with col1:
                column_name = st.text_input(
                    f"Column {i + 1} Name:",
                    value=st.session_state.get(f"gen_col_name_{i}", f"Column_{i + 1}"),
                    key=f"generation_column_name_{i}",
                    help=f"Name for column {i + 1}",
                )

            with col2:
                column_desc = st.text_area(
                    f"Column {i + 1} Description:",
                    value=st.session_state.get(f"gen_col_desc_{i}", ""),
                    height=80,
                    key=f"generation_column_desc_{i}",
                    help="Describe what this column should contain",
                )

            if column_name.strip():
                generation_columns[column_name] = i
                column_descriptions[column_name] = column_desc.strip()
                st.session_state[f"gen_col_name_{i}"] = column_name
                st.session_state[f"gen_col_desc_{i}"] = column_desc

        if not generation_columns:
            st.info("Please define at least one output column.")
            return None

        # Additional generation parameters
        st.markdown("### **Advanced Generation Parameters**")

        col1, col2 = st.columns(2)

        with col1:
            temperature = st.slider(
                "Temperature (creativity):",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.get("generation_temperature", 0.7),
                step=0.1,
                key="generation_temperature_slider",
                help="Higher values make output more creative but less consistent",
            )

        with col2:
            max_tokens = st.number_input(
                "Max tokens per generation:",
                min_value=50,
                max_value=2000,
                value=st.session_state.get("generation_max_tokens", 500),
                step=50,
                key="generation_max_tokens_input",
                help="Maximum tokens for each generated item",
            )

        # Build configuration dictionary
        config = {
            "generation_prompt": generation_prompt,
            "columns": generation_columns,
            "column_descriptions": column_descriptions,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Store in app instance and session state
        app_instance.generation_config = config
        st.session_state["generation_config"] = config
        st.session_state["generation_prompt"] = generation_prompt
        st.session_state["generation_temperature"] = temperature
        st.session_state["generation_max_tokens"] = max_tokens

        # Show configuration summary
        st.markdown("### **Configuration Summary**")
        st.info(f"📝 Configuration ready with **{len(generation_columns)} columns**")

        # Use checkbox instead of nested expander
        show_setup = st.checkbox(
            "Show Generation Setup Details", value=False, key="show_gen_setup"
        )
        if show_setup:
            st.write("**Columns to generate:**")
            for col_name, desc in column_descriptions.items():
                st.write(f"- **{col_name}**: {desc}")

            st.write("**Generation Parameters:**")
            st.write(f"- Temperature: {temperature}")
            st.write(f"- Max tokens: {max_tokens}")

        return config
