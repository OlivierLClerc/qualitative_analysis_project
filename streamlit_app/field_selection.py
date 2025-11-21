"""
Module for handling field selection functionality in the Streamlit app.
"""

import streamlit as st
from typing import Any, List
from streamlit_app.session_management import save_session


def select_fields(app_instance: Any, step_number: int = 4) -> List[str]:
    """
    Fields to Extract
    Allows the user to specify which fields (e.g., 'Evaluation', 'Comments')
    the LLM should return in its JSON output.
    Also asks which field is the label column when annotation columns are available.

    Args:
        app_instance: The QualitativeAnalysisApp instance
        step_number: The step number to display (default=4 for annotation mode)

    Returns:
        A list of selected fields
    """
    st.markdown(f"### Step {step_number}: Fields to Extract", unsafe_allow_html=True)
    with st.expander(f"Show/hide details of step {step_number}", expanded=True):
        st.markdown(
            """
            Specify the **fields** (or categories) you want the model to generate for each entry.
            The names should match the field names used in your codebook & examples (if provided).
            """,
            unsafe_allow_html=True,
        )

        default_fields = (
            ",".join(app_instance.selected_fields)
            if app_instance.selected_fields
            else ""
        )
        fields_str = st.text_input(
            "Comma-separated fields (e.g. 'Reasoning, Classification')",
            value=default_fields,
            key="fields_input",
        )
        extracted = [f.strip() for f in fields_str.split(",") if f.strip()]

        # Update app instance and session state
        app_instance.selected_fields = extracted
        st.session_state["selected_fields"] = extracted

        # -------- Types and optional enums ---------
        field_types_existing = getattr(app_instance, "field_types", {}) or st.session_state.get(
            "field_types", {}
        )
        field_enums_existing = getattr(app_instance, "field_enums", {}) or st.session_state.get(
            "field_enums", {}
        )

        field_types = {}
        field_enums = {}

        if extracted:
            st.subheader("Field Types and Allowed Values")
            st.markdown(
                """
                For each field, choose its **type**, and optionally define a **set of allowed values**
                (e.g., *1, 2, 3, 4, 5* or *low, medium, high*).
                """
            )
            type_options = ["string", "number", "boolean"]

            for field in extracted:
                with st.container(border=True):
                    st.markdown(f"**{field}**")
                    col1, col2 = st.columns([2, 4])
                    # Field type selection
                    default_type = field_types_existing.get(field, "string")
                    with col1:
                        selected_type = st.selectbox(
                            f"Type",
                            options=type_options,
                            index=type_options.index(default_type) if default_type in type_options else 0,
                            help="Choose **string** for text, **number** for numbers, and **boolean** for a binary value such as True/False.",
                            key=f"field_type_{field}",
                        )
                    field_types[field] = selected_type

                    # Whether this field uses a list of allowed values
                    existing_enum = field_enums_existing.get(field, None)
                    has_enum_default = existing_enum is not None and len(existing_enum) > 0
                    with col2:
                        if selected_type == "boolean":
                            use_enum = st.checkbox(
                            f"Restrict '{field}' to a list of allowed values?",
                            value=has_enum_default,
                            disabled=True,
                            key=f"field_use_enum_{field}",
                        )
                        else :
                            use_enum = st.checkbox(
                                f"Restrict '{field}' to a list of allowed values?",
                                value=has_enum_default,
                                key=f"field_use_enum_{field}",
                            )

                    if use_enum:
                        default_enum_str = (
                            ", ".join(str(v) for v in existing_enum)
                            if existing_enum
                            else ""
                        )
                        with col2:
                            enum_str = st.text_input(
                                f"Allowed values for '{field}' (comma-separated):",
                                value=default_enum_str,
                                key=f"field_enum_values_{field}",
                            )
                        raw_values = [v.strip() for v in enum_str.split(",") if v.strip()]

                        # Cast according to the field type
                        if selected_type == "number":
                            cast_values = []
                            for v in raw_values:
                                try:
                                    cast_values.append(float(v))
                                except ValueError:
                                    # If casting fails, keep as string
                                    cast_values.append(v)
                            field_enums[field] = cast_values
                        else:
                            field_enums[field] = raw_values
                    else:
                        field_enums[field] = []

            # Save in state
            app_instance.field_types = field_types
            app_instance.field_enums = field_enums
            st.session_state["field_types"] = field_types
            st.session_state["field_enums"] = field_enums

        # Only show label selection if fields have been specified and annotation columns exist
        if app_instance.selected_fields and app_instance.annotation_columns:
            st.subheader("Label Column Selection")
            st.markdown(
                """
                Select which field is your **label column** (the main classification or prediction)
                that should match the format of your annotation columns.
                """
            )

            # Get the default index for the label column
            default_index = 0
            if (
                app_instance.label_column
                and app_instance.label_column in app_instance.selected_fields
            ):
                default_index = app_instance.selected_fields.index(
                    app_instance.label_column
                )
            elif not app_instance.selected_fields:
                default_index = None

            # Select the label column from the extracted fields
            label_column = st.selectbox(
                "Label Column:",
                options=app_instance.selected_fields,
                index=default_index,
                key="label_column_select",
            )

            # Store in session state and app instance
            st.session_state["label_column"] = label_column
            app_instance.label_column = label_column

        # Add the save session functionality at the end of Step 4
        save_session(app_instance)

    return extracted
