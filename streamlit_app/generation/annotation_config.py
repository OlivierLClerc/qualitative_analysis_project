"""
Module for handling annotation configuration in the generation workflow.
"""

import streamlit as st
from typing import Any, Dict, Optional


def configure_annotation(app_instance: Any) -> Optional[Dict[str, Any]]:
    """
    Step 4: Annotation Configuration
    Allows users to configure how to annotate the generated content.

    Args:
        app_instance: The QualitativeAnalysisApp instance

    Returns:
        Dictionary containing annotation configuration or None if incomplete
    """
    st.markdown("### Step 4: Annotation Configuration", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 4", expanded=True):

        # Check if generated content is available
        if (
            not hasattr(app_instance, "generated_content")
            or app_instance.generated_content is None
        ):
            st.warning("Please generate content in Step 3 first.")
            return None

        st.markdown(
            """
            ### **Annotation Setup**
            Configure how the LLM should annotate the generated content.
            This step will add annotation columns to your generated dataset.
            """,
            unsafe_allow_html=True,
        )

        # Annotation prompt
        default_annotation_prompt = st.session_state.get("annotation_prompt", "")
        annotation_prompt = st.text_area(
            "**Annotation Prompt:**",
            value=default_annotation_prompt,
            height=200,
            key="annotation_prompt_textarea",
            help="Describe how to annotate the generated content (e.g., 'Verify similarity with blueprint exercise, validate correctness of answers, assess difficulty level')",
        )

        if not annotation_prompt.strip():
            st.info("Please provide an annotation prompt to proceed.")
            return None

        st.markdown("### **Annotation Columns Configuration**")
        st.markdown(
            """
            Define what annotation fields to add to your generated content.
            These will be new columns added to each row of generated content.
            """,
            unsafe_allow_html=True,
        )

        # Column configuration
        num_annotation_columns = st.number_input(
            "Number of annotation columns:",
            min_value=1,
            max_value=15,
            value=st.session_state.get("num_annotation_columns", 3),
            step=1,
            key="num_annotation_columns_input",
        )

        st.session_state["num_annotation_columns"] = num_annotation_columns

        # Convert to int for range usage
        num_annotation_columns_int = int(num_annotation_columns)

        annotation_columns: Dict[str, int] = {}
        annotation_column_descriptions: Dict[str, str] = {}

        for i in range(num_annotation_columns_int):
            col1, col2 = st.columns([1, 2])

            with col1:
                column_name = st.text_input(
                    f"Annotation Column {i + 1} Name:",
                    value=st.session_state.get(
                        f"ann_col_name_{i}", f"Annotation_{i + 1}"
                    ),
                    key=f"annotation_column_name_{i}",
                    help=f"Name for annotation column {i + 1}",
                )

            with col2:
                column_desc = st.text_area(
                    f"Annotation Column {i + 1} Description:",
                    value=st.session_state.get(f"ann_col_desc_{i}", ""),
                    height=80,
                    key=f"annotation_column_desc_{i}",
                    help="Describe what this annotation column should contain",
                )

            if column_name.strip():
                annotation_columns[column_name] = i
                annotation_column_descriptions[column_name] = column_desc.strip()
                st.session_state[f"ann_col_name_{i}"] = column_name
                st.session_state[f"ann_col_desc_{i}"] = column_desc

        if not annotation_columns:
            st.info("Please define at least one annotation column.")
            return None

        # Additional annotation parameters
        st.markdown("### **Advanced Annotation Parameters**")

        col1, col2 = st.columns(2)

        with col1:
            annotation_temperature = st.slider(
                "Annotation Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("annotation_temperature", 0.0),
                step=0.1,
                key="annotation_temperature_slider",
                help="Lower values make annotations more consistent (recommended: 0.0-0.3)",
            )

        with col2:
            annotation_max_tokens = st.number_input(
                "Max tokens per annotation:",
                min_value=50,
                max_value=1000,
                value=st.session_state.get("annotation_max_tokens", 300),
                step=50,
                key="annotation_max_tokens_input",
                help="Maximum tokens for each annotation response",
            )

        # Show generated content summary for context
        st.markdown("### **Generated Content Summary**")
        generated_df = app_instance.generated_content
        generation_columns = [
            col for col in generated_df.columns if col != "generation_id"
        ]

        st.info(
            f"📊 Generated content has {len(generated_df)} items with {len(generation_columns)} content columns"
        )

        # Show as collapsible section without nested expander
        show_preview = st.checkbox(
            "Show Generated Content Structure",
            value=False,
            key="show_gen_content_structure",
        )
        if show_preview:
            st.write("**Content columns:**")
            for col in generation_columns:
                st.write(f"- **{col}**")

            st.write("**Sample row:**")
            if len(generated_df) > 0:
                sample_row = generated_df.iloc[0]
                for col in generation_columns:
                    content = str(sample_row[col])
                    preview_content = (
                        content[:100] + "..." if len(content) > 100 else content
                    )
                    st.write(f"- **{col}**: {preview_content}")

        # Build configuration dictionary
        config = {
            "annotation_prompt": annotation_prompt,
            "annotation_columns": annotation_columns,
            "annotation_column_descriptions": annotation_column_descriptions,
            "annotation_temperature": annotation_temperature,
            "annotation_max_tokens": annotation_max_tokens,
        }

        # Store in app instance and session state
        app_instance.annotation_config = config
        st.session_state["annotation_config"] = config
        st.session_state["annotation_prompt"] = annotation_prompt
        st.session_state["annotation_temperature"] = annotation_temperature
        st.session_state["annotation_max_tokens"] = annotation_max_tokens

        # Show configuration summary
        st.markdown("### **Annotation Configuration Summary**")
        st.info(
            f"📝 Will annotate **{len(generated_df)} items** with **{len(annotation_columns)} annotation columns**"
        )

        # Use checkbox instead of nested expander
        show_annotation_setup = st.checkbox(
            "Show Annotation Setup Details", value=False, key="show_ann_setup"
        )
        if show_annotation_setup:
            st.write("**Annotation columns to add:**")
            for col_name, desc in annotation_column_descriptions.items():
                st.write(f"- **{col_name}**: {desc}")

            st.write("**Annotation Parameters:**")
            st.write(f"- Temperature: {annotation_temperature}")
            st.write(f"- Max tokens: {annotation_max_tokens}")
            st.write(f"- Items to annotate: {len(generated_df)}")

        return config
