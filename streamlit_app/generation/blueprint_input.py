"""
Module for handling blueprint/reference input functionality in the generation workflow.
"""

import streamlit as st
from typing import Any, Optional, Dict


def blueprint_input(app_instance: Any) -> Optional[Dict[str, str]]:
    """
    Step 1: Blueprint/Reference Input
    Allows users to provide reference examples that will guide the content generation.

    Args:
        app_instance: The QualitativeAnalysisApp instance

    Returns:
        List of blueprint texts or None if no blueprints were provided
    """
    st.markdown("### Step 1: Blueprint/Reference Input", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 1", expanded=True):
        st.markdown(
            """
            ### **Blueprint Instructions**
            Provide one or more reference examples that will guide the content generation.
            These examples serve as blueprints for the LLM to understand the format, style, and content type you want to generate.
            
            **Example Use Cases:**
            - **Exercise Generation**: Provide a sample exercise to generate similar ones with variations
            - **Question Generation**: Give example questions to create more in the same style
            - **Content Creation**: Show template content to generate variations
            """,
            unsafe_allow_html=True,
        )

        # Multiple blueprint input methods
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "File Upload"],
            key="blueprint_input_method",
        )

        blueprints_dict = {}

        if input_method == "Text Input":
            # Allow multiple blueprints via text input
            num_blueprints = st.number_input(
                "Number of blueprint examples:",
                min_value=1,
                max_value=10,
                value=st.session_state.get("num_blueprints", 1),
                key="num_blueprints_input",
            )

            st.session_state["num_blueprints"] = num_blueprints

            for i in range(num_blueprints):
                blueprint_text = st.text_area(
                    f"**Blueprint Example {i + 1}:**",
                    value=st.session_state.get(f"blueprint_{i}", ""),
                    height=200,
                    key=f"blueprint_text_{i}",
                    help=f"Enter your reference example #{i + 1} here",
                )

                if blueprint_text.strip():
                    blueprints_dict[f"Blueprint_{i + 1}"] = blueprint_text.strip()
                    st.session_state[f"blueprint_{i}"] = blueprint_text.strip()

        elif input_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload blueprint file (TXT):",
                type=["txt"],
                key="blueprint_file_upload",
            )

            if uploaded_file is not None:
                try:
                    # Read the file content
                    blueprint_content = uploaded_file.read().decode("utf-8")

                    # Option to split by delimiter or treat as single blueprint
                    split_option = st.radio(
                        "File processing:",
                        [
                            "Single blueprint",
                            "Multiple blueprints (split by delimiter)",
                        ],
                        key="file_split_option",
                    )

                    if split_option == "Single blueprint":
                        blueprints_dict["Blueprint_1"] = blueprint_content.strip()
                        st.text_area(
                            "Preview:",
                            value=blueprint_content,
                            height=200,
                            disabled=True,
                        )
                    else:
                        delimiter = st.text_input(
                            "Delimiter to split blueprints:",
                            value="---",
                            key="blueprint_delimiter",
                        )

                        split_blueprints = blueprint_content.split(delimiter)
                        split_blueprints = [
                            bp.strip() for bp in split_blueprints if bp.strip()
                        ]

                        for i, bp in enumerate(split_blueprints):
                            blueprints_dict[f"Blueprint_{i + 1}"] = bp

                        st.write(f"Found {len(split_blueprints)} blueprint examples:")
                        for i, bp in enumerate(
                            split_blueprints[:3]
                        ):  # Show first 3 as preview
                            st.markdown(f"**Preview Blueprint {i + 1}:**")
                            preview_text = bp[:500] + "..." if len(bp) > 500 else bp
                            st.text_area(
                                f"Preview {i + 1}",
                                value=preview_text,
                                height=100,
                                disabled=True,
                                key=f"file_blueprint_preview_{i}",
                                label_visibility="collapsed",
                            )

                        if len(split_blueprints) > 3:
                            st.info(
                                f"... and {len(split_blueprints) - 3} more blueprints"
                            )

                except Exception as e:
                    st.error(f"Error reading file: {e}")

        # Store blueprints in app instance and session state
        if blueprints_dict:
            app_instance.blueprints = blueprints_dict
            st.session_state["blueprints"] = blueprints_dict

            st.success(f"✅ {len(blueprints_dict)} blueprint(s) loaded successfully!")

            # Show summary
            st.subheader("Blueprint Summary")
            for col_name, blueprint in blueprints_dict.items():
                st.markdown(f"**{col_name}** ({len(blueprint)} characters)")
                preview_text = (
                    blueprint[:300] + "..." if len(blueprint) > 300 else blueprint
                )
                st.text_area(
                    f"Preview {col_name}:",
                    value=preview_text,
                    height=100,
                    disabled=True,
                    key=f"blueprint_preview_{col_name}",
                    label_visibility="collapsed",
                )
        else:
            st.info("Please provide at least one blueprint example to proceed.")
            return None

    return blueprints_dict if blueprints_dict else None
