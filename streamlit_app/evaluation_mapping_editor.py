"""
Shared UI for configuring evaluation mappings across Streamlit workflows.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import streamlit as st

from streamlit_app.evaluation_mappings import (
    LABEL_TYPE_OPTIONS,
    EvaluationMapping,
    clear_evaluation_result_cache,
    create_default_mapping,
    sanitize_evaluation_mappings,
)


def _serialize_mappings(mappings: Sequence[Mapping[str, Any]]) -> List[tuple[Any, ...]]:
    return [
        (
            mapping.get("id"),
            mapping.get("name"),
            mapping.get("llm_field"),
            tuple(mapping.get("human_columns", [])),
            mapping.get("label_type"),
        )
        for mapping in mappings
    ]


def _widget_key(prefix: str, key: str) -> str:
    return f"{prefix}{key}"


def _sync_widget_state(
    *,
    widget_key: str,
    default_value: Any,
    valid_options: Optional[Sequence[Any]] = None,
    allow_multiple: bool = False,
) -> None:
    """
    Initialize or sanitize widget session state without overwriting active user edits.
    """
    if widget_key not in st.session_state:
        st.session_state[widget_key] = default_value
        return

    if valid_options is None:
        return

    valid_set = set(valid_options)
    current_value = st.session_state[widget_key]

    if allow_multiple:
        filtered_value = [value for value in current_value if value in valid_set]
        if filtered_value != current_value:
            st.session_state[widget_key] = filtered_value
    elif current_value not in valid_set:
        st.session_state[widget_key] = default_value


def render_evaluation_mapping_editor(
    *,
    selected_fields: Sequence[str],
    annotation_columns: Sequence[str],
    raw_mappings: Optional[Sequence[Mapping[str, Any]]] = None,
    legacy_label_column: Optional[str] = None,
    legacy_label_type: Optional[str] = None,
    mapping_state_key: str = "evaluation_mappings",
    initialized_state_key: str = "evaluation_mappings_initialized",
    ui_key_prefix: str = "",
    llm_field_label: str = "LLM field",
    intro_markdown: Optional[str] = None,
    create_default_if_empty: bool = True,
) -> List[EvaluationMapping]:
    """
    Render the reusable evaluation mapping editor and persist its state.
    """
    if intro_markdown:
        st.markdown(intro_markdown, unsafe_allow_html=True)

    if not selected_fields or not annotation_columns:
        st.session_state[mapping_state_key] = []
        st.session_state[initialized_state_key] = False
        return []

    mappings_initialized = st.session_state.get(initialized_state_key, False)
    mappings = sanitize_evaluation_mappings(
        raw_mappings=raw_mappings,
        selected_fields=selected_fields,
        annotation_columns=annotation_columns,
        legacy_label_column=legacy_label_column,
        legacy_label_type=legacy_label_type,
        create_default_if_empty=create_default_if_empty and not mappings_initialized,
    )

    updated_mappings: List[EvaluationMapping] = []
    removed_mapping_id = None

    for index, mapping in enumerate(mappings):
        st.markdown(f"**Mapping {index + 1}**")
        col1, col2, col3, col4, col5 = st.columns([1.4, 1.2, 1.8, 1.0, 0.6])
        name_key = _widget_key(
            ui_key_prefix, f"evaluation_mapping_name_{mapping['id']}"
        )
        llm_key = _widget_key(ui_key_prefix, f"evaluation_mapping_llm_{mapping['id']}")
        humans_key = _widget_key(
            ui_key_prefix, f"evaluation_mapping_humans_{mapping['id']}"
        )
        type_key = _widget_key(
            ui_key_prefix, f"evaluation_mapping_type_{mapping['id']}"
        )

        _sync_widget_state(widget_key=name_key, default_value=mapping["name"])
        _sync_widget_state(
            widget_key=llm_key,
            default_value=mapping["llm_field"],
            valid_options=selected_fields,
        )
        _sync_widget_state(
            widget_key=humans_key,
            default_value=[
                column
                for column in mapping["human_columns"]
                if column in annotation_columns
            ],
            valid_options=annotation_columns,
            allow_multiple=True,
        )
        _sync_widget_state(
            widget_key=type_key,
            default_value=mapping["label_type"],
            valid_options=LABEL_TYPE_OPTIONS,
        )

        with col1:
            mapping_name = st.text_input(
                "Display name",
                key=name_key,
            )

        with col2:
            llm_field = st.selectbox(
                llm_field_label,
                options=list(selected_fields),
                key=llm_key,
            )

        with col3:
            human_columns = st.multiselect(
                "Human columns",
                options=list(annotation_columns),
                key=humans_key,
            )

        with col4:
            label_type = st.selectbox(
                "Label type",
                options=LABEL_TYPE_OPTIONS,
                key=type_key,
            )

        with col5:
            remove_clicked = st.button(
                "Remove",
                key=_widget_key(
                    ui_key_prefix, f"evaluation_mapping_remove_{mapping['id']}"
                ),
            )

        if remove_clicked:
            removed_mapping_id = mapping["id"]
        else:
            updated_mappings.append(
                {
                    "id": mapping["id"],
                    "name": mapping_name.strip() or llm_field,
                    "llm_field": llm_field,
                    "human_columns": human_columns,
                    "label_type": label_type,
                }
            )

        if not human_columns and not remove_clicked:
            st.warning(
                f"Mapping `{mapping_name.strip() or llm_field}` needs at least one human annotation column."
            )

        st.markdown("---")

    if st.button(
        "Add evaluation mapping",
        key=_widget_key(ui_key_prefix, "add_evaluation_mapping_button"),
    ):
        updated_mappings.append(
            create_default_mapping(
                selected_fields=selected_fields,
                annotation_columns=annotation_columns,
                index=len(updated_mappings),
            )
        )
        st.session_state[mapping_state_key] = updated_mappings
        st.session_state[initialized_state_key] = True
        clear_evaluation_result_cache(st.session_state, prefix=ui_key_prefix)
        st.rerun()

    if removed_mapping_id is not None:
        st.session_state[mapping_state_key] = updated_mappings
        st.session_state[initialized_state_key] = True
        clear_evaluation_result_cache(st.session_state, prefix=ui_key_prefix)
        st.rerun()

    mappings_changed = _serialize_mappings(updated_mappings) != _serialize_mappings(
        st.session_state.get(mapping_state_key, [])
    )

    st.session_state[mapping_state_key] = updated_mappings
    st.session_state[initialized_state_key] = True

    if mappings_changed:
        clear_evaluation_result_cache(st.session_state, prefix=ui_key_prefix)

    return updated_mappings
