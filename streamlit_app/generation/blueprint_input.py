"""
Module for handling blueprint/reference input functionality in the generation workflow.
"""

import streamlit as st
import json
from typing import Any, Optional, Dict

from streamlit_app.gameplay_utils import (
    is_gameplay_config,
    get_available_gameplays,
    get_gameplay_description,
)


def blueprint_input(app_instance: Any) -> Optional[Dict[str, str]]:
    """
    Step 1: Reference Input
    Allows users to provide reference examples that will guide the content generation.

    Args:
        app_instance: The QualitativeAnalysisApp instance

    Returns:
        List of blueprint texts or None if no blueprints were provided
    """
    st.markdown("### Step 1: Blueprint/Reference Input", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 1", expanded=True):
        # Check for gameplay mode first
        st.markdown("---")
        st.markdown("### Gameplay Mode (Optional)")
        use_gameplay = st.checkbox(
            "Use gameplay template configuration?",
            value=st.session_state.get("generation_use_gameplay", False),
            key="generation_gameplay_checkbox",
            help="Load blueprint and settings from a gameplay configuration template",
        )

        st.session_state["generation_use_gameplay"] = use_gameplay

        if use_gameplay:
            st.markdown(
                """
            Upload a **gameplay configuration JSON** that contains generation templates 
            and settings for different exercise types.
            """
            )

            gameplay_config_file = st.file_uploader(
                "Upload Gameplay Config (JSON)",
                type=["json"],
                key="generation_gameplay_config_uploader",
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
                    st.session_state["generation_gameplay_config"] = gameplay_config

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
                        "generation_selected_gameplay" in st.session_state
                        and st.session_state["generation_selected_gameplay"]
                        in available_gameplays
                    ):
                        default_index = available_gameplays.index(
                            st.session_state["generation_selected_gameplay"]
                        )

                    selected_gameplay = st.selectbox(
                        "Choose which gameplay type to generate:",
                        options=available_gameplays,
                        index=default_index,
                        key="generation_gameplay_selector",
                    )

                    # Show gameplay description
                    gameplay_desc = get_gameplay_description(
                        gameplay_config, selected_gameplay
                    )
                    if gameplay_desc:
                        st.info(f"ℹ️ **{selected_gameplay}**: {gameplay_desc}")

                    # Get blueprint from gameplay config
                    gameplay_data = gameplay_config["gameplays"].get(
                        selected_gameplay, {}
                    )
                    blueprint_from_config = gameplay_data.get(
                        "blueprint_example", gameplay_data.get("blueprint_ref", "")
                    )

                    if blueprint_from_config:
                        st.success("✅ Blueprint loaded from gameplay template")

                        # Store everything
                        st.session_state["generation_selected_gameplay"] = (
                            selected_gameplay
                        )
                        st.session_state["generation_gameplay_data"] = gameplay_data

                        # Pre-populate blueprint
                        blueprints_dict = {"Blueprint_1": blueprint_from_config}
                        app_instance.blueprints = blueprints_dict
                        st.session_state["blueprints"] = blueprints_dict
                        st.session_state["blueprint_0"] = blueprint_from_config
                        st.session_state["num_blueprints"] = 1

                    else:
                        st.warning(
                            "⚠️ No blueprint found in the selected gameplay config. Please enter one manually below."
                        )

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
                st.markdown("---")

        # Normal blueprint input section
        st.markdown("---")
        st.markdown(
            """
            ### **Instructions**
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
