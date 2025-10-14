"""
Module for handling field selection functionality in the Streamlit app.
"""

import streamlit as st
from typing import Any, List, Dict
from streamlit_app.session_management import save_session


def select_fields(app_instance: Any, step_number: int = 4) -> List[Dict[str, str]]:
    """
    Fields to Extract
    Allows the user to specify which fields (e.g., 'Evaluation', 'Comments')
    the LLM should return in its JSON output.
    Also asks which field is the label column.

    Args:
        app_instance: The QualitativeAnalysisApp instance
        step_number: The step number to display (default=4 for annotation mode)

    Returns:
        A list of field dictionaries, each containing 'name' and 'description'
    """
    st.markdown(f"### Step {step_number}: Fields to Extract", unsafe_allow_html=True)
    with st.expander(f"Show/hide details of step {step_number}", expanded=True):
        # Check if we're in gameplay mode with a loaded config for pre-population
        use_gameplay = st.session_state.get("use_gameplay_mode", False)
        merged_config = st.session_state.get("gameplay_merged_config", None)

        # Pre-populate defaults from gameplay config if available
        if use_gameplay and merged_config:
            st.info(
                f"**Gameplay Mode Active** - Fields pre-populated from template for gameplay: {st.session_state.get('selected_gameplay', 'Unknown')}"
            )
            st.markdown("*You can review and modify all pre-filled values below.*")

            # Pre-populate selected_fields from gameplay config
            gameplay_fields = merged_config.get("selected_fields", [])

            # Ensure fields are in the new dict format with name and description
            formatted_fields = []
            for field in gameplay_fields:
                if isinstance(field, dict):
                    formatted_fields.append(field)
                else:
                    # Convert old string format to new dict format
                    formatted_fields.append({"name": field, "description": ""})

            st.session_state["selected_fields"] = formatted_fields
            app_instance.selected_fields = formatted_fields

            # Pre-populate individual field values in session state for the UI
            for i, field in enumerate(formatted_fields):
                st.session_state[f"annotation_field_name_{i}"] = field["name"]
                st.session_state[f"annotation_field_desc_{i}"] = field.get(
                    "description", ""
                )

            # Set the number of fields
            st.session_state["num_annotation_fields"] = len(formatted_fields)

        st.markdown(
            """
            Specify the **fields** (or categories) you want the model to generate for each entry.
            The names should match the field names used in your codebook & examples (if provided).
            """,
            unsafe_allow_html=True,
        )

        # Determine default number of fields
        default_num_fields = 2
        if app_instance.selected_fields:
            if (
                isinstance(app_instance.selected_fields, list)
                and len(app_instance.selected_fields) > 0
            ):
                if isinstance(app_instance.selected_fields[0], dict):
                    default_num_fields = len(app_instance.selected_fields)
                else:
                    default_num_fields = len(app_instance.selected_fields)

        # Number of fields input
        num_fields = st.number_input(
            "Number of fields to extract:",
            min_value=1,
            max_value=10,
            value=st.session_state.get("num_annotation_fields", default_num_fields),
            step=1,
            key="num_annotation_fields_input",
        )

        st.session_state["num_annotation_fields"] = num_fields

        # Convert to int for range usage
        num_fields_int = int(num_fields)

        extracted = []
        field_descriptions = {}

        st.markdown("### **Field Configuration**")

        for i in range(num_fields_int):
            col1, col2 = st.columns([1, 2])

            with col1:
                # Get default field name
                default_name = f"Field_{i + 1}"
                if app_instance.selected_fields and i < len(
                    app_instance.selected_fields
                ):
                    if isinstance(app_instance.selected_fields[i], dict):
                        default_name = app_instance.selected_fields[i].get(
                            "name", default_name
                        )
                    else:
                        default_name = app_instance.selected_fields[i]

                field_name = st.text_input(
                    f"Field {i + 1} Name:",
                    value=st.session_state.get(
                        f"annotation_field_name_{i}", default_name
                    ),
                    key=f"annotation_field_name_{i}",
                    help=f"Name for field {i + 1} (e.g., 'Reasoning', 'Classification')",
                )

            with col2:
                # Get default description
                default_desc = ""
                if app_instance.selected_fields and i < len(
                    app_instance.selected_fields
                ):
                    if isinstance(app_instance.selected_fields[i], dict):
                        default_desc = app_instance.selected_fields[i].get(
                            "description", ""
                        )

                field_desc = st.text_area(
                    f"Field {i + 1} Description:",
                    value=st.session_state.get(
                        f"annotation_field_desc_{i}", default_desc
                    ),
                    height=80,
                    key=f"annotation_field_desc_{i}",
                    help="Describe what this field should contain",
                )

            if field_name.strip():
                extracted.append(
                    {"name": field_name.strip(), "description": field_desc.strip()}
                )
                field_descriptions[field_name.strip()] = field_desc.strip()

        # Update app instance and session state
        app_instance.selected_fields = extracted
        st.session_state["selected_fields"] = extracted

        # Only show label selection if fields have been specified and annotation columns exist
        if app_instance.selected_fields and app_instance.annotation_columns:
            st.subheader("Label Column Selection")
            st.markdown(
                """
                Select which field is your **label column** (the main classification or prediction)
                that should match the format of your annotation columns.
                """
            )

            # Extract field names for selection
            field_names = [field["name"] for field in app_instance.selected_fields]

            # Get the default index for the label column
            default_index = 0
            if app_instance.label_column and app_instance.label_column in field_names:
                default_index = field_names.index(app_instance.label_column)

            # Select the label column from the extracted fields
            label_column = st.selectbox(
                "Label Column:",
                options=field_names,
                index=default_index,
                key="label_column_select",
            )

            # Store in session state and app instance
            st.session_state["label_column"] = label_column
            app_instance.label_column = label_column

        # Add the save session functionality at the end of Step 4
        save_session(app_instance)

    return extracted
