"""
Dedicated Streamlit page for computing metrics from existing annotation columns.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Pattern, Sequence, TypedDict

import numpy as np
import pandas as pd
import streamlit as st

from qualitative_analysis.metrics.kappa import compute_cohens_kappa
from qualitative_analysis.metrics.utils import compute_majority_vote
from streamlit_app.evaluation_dashboard import (
    KAPPA_METHOD,
    render_evaluation_dashboard,
)
from streamlit_app.evaluation_mappings import (
    EvaluationMapping,
    KAPPA_WEIGHT_OPTIONS,
    KRIPP_LEVEL_OPTIONS,
    LABEL_TYPE_OPTIONS,
    clear_evaluation_result_cache,
    create_mapping_id,
    default_kripp_level_for_label_type,
)


METRICS_ONLY_PREFIX = "metrics_only_"
DEFAULT_COLUMN_PATTERN_TEMPLATE = "Rater_{rater}_{scale}"
SUPPORTED_PATTERN_PLACEHOLDERS = {"rater", "scale"}


class ColumnPatternMetadata(TypedDict, total=False):
    scale: str
    rater: str


class ColumnPatternParseResult(TypedDict):
    template: str
    error: Optional[str]
    placeholders: List[str]
    column_metadata: Dict[str, ColumnPatternMetadata]
    matched_columns: List[str]
    unmatched_columns: List[str]
    matched_scale_groups: Dict[str, List[str]]
    skipped_scales: Dict[str, List[str]]
    preview_rows: List[Dict[str, str]]
    supports_rater: bool


def _state_key(name: str) -> str:
    return f"{METRICS_ONLY_PREFIX}{name}"


def _clear_metrics_only_cached_results() -> None:
    clear_evaluation_result_cache(st.session_state, prefix=METRICS_ONLY_PREFIX)
    st.session_state.pop(_state_key("scale_correlation_results"), None)
    st.session_state.pop(_state_key("annotator_agreement_results"), None)


def _reset_metrics_only_state(data: pd.DataFrame, upload_signature: tuple) -> None:
    st.session_state[_state_key("data")] = data
    st.session_state[_state_key("uploaded_dataset_signature")] = upload_signature
    st.session_state[_state_key("annotation_columns")] = []
    st.session_state[_state_key("annotation_columns_selection")] = []
    st.session_state.pop(_state_key("applied_column_pattern_template"), None)
    st.session_state[_state_key("evaluation_mappings")] = []
    st.session_state[_state_key("evaluation_mappings_initialized")] = False
    _clear_metrics_only_cached_results()


def _upload_metrics_dataset() -> Optional[pd.DataFrame]:
    from streamlit_app.data_upload import build_upload_signature, load_uploaded_dataset

    st.markdown("### Step 1: Upload a Dataset", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 1", expanded=True):
        st.markdown(
            """
            Upload a dataset that already contains annotation columns you want to evaluate.
            Optional metadata columns such as `run`, `split`, `iteration`, `prompt_name`,
            or `prompt_iteration` are preserved when present.
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload CSV or XLSX",
            type=["csv", "xlsx"],
            key=_state_key("dataset_uploader"),
        )
        if uploaded_file is None:
            return st.session_state.get(_state_key("data"))

        delimiter = st.text_input(
            "CSV Delimiter (if CSV)",
            value=st.session_state.get(_state_key("delimiter"), ";"),
            key=_state_key("delimiter"),
        )
        upload_signature = build_upload_signature(uploaded_file, delimiter)
        previous_signature = st.session_state.get(
            _state_key("uploaded_dataset_signature")
        )
        should_reload_data = (
            previous_signature != upload_signature
            or st.session_state.get(_state_key("data")) is None
        )

        if should_reload_data:
            try:
                data = load_uploaded_dataset(uploaded_file, delimiter)
                _reset_metrics_only_state(data, upload_signature)
                st.success("Data loaded successfully!")
            except Exception as exc:
                st.error(f"Error loading data: {exc}")
                st.stop()

        data = st.session_state.get(_state_key("data"))
        if data is not None:
            st.write("Data Preview:", data.head())
        return data


def _normalize_missing_value(value: Any) -> Any:
    if value is None:
        return pd.NA

    try:
        if pd.isna(value):
            return pd.NA
    except TypeError:
        pass

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "nan", "<na>", "n/a", "null"}:
            return pd.NA
        return stripped

    return value


def infer_label_type_for_columns(data: pd.DataFrame, columns: Sequence[str]) -> str:
    """
    Infer whether a mapping behaves like Integer, Float, or Text labels.
    """
    observed_types = set()

    for column in columns:
        if column not in data.columns:
            continue

        for raw_value in data[column].tolist():
            value = _normalize_missing_value(raw_value)
            if value is pd.NA:
                continue

            if isinstance(value, bool):
                observed_types.add("Integer")
                continue

            if isinstance(value, int):
                observed_types.add("Integer")
                continue

            if isinstance(value, float):
                observed_types.add("Integer" if float(value).is_integer() else "Float")
                continue

            text_value = str(value).replace(",", ".")
            match = re.search(r"[-+]?\d+(?:\.\d+)?", text_value)
            if not match:
                observed_types.add("Text")
                continue

            numeric_value = float(match.group(0))
            observed_types.add("Integer" if numeric_value.is_integer() else "Float")

    if not observed_types:
        return "Text"
    if observed_types == {"Integer"}:
        return "Integer"
    if observed_types.issubset({"Integer", "Float"}):
        return "Float" if "Float" in observed_types else "Integer"
    return "Text"


def _compile_column_pattern_template(
    template: str,
) -> tuple[Optional[Pattern[str]], List[str], Optional[str]]:
    cleaned_template = template.strip()
    if not cleaned_template:
        return None, [], "Column pattern template cannot be empty."

    regex_parts: List[str] = []
    placeholders: List[str] = []
    cursor = 0
    previous_was_placeholder = False

    while cursor < len(cleaned_template):
        current_char = cleaned_template[cursor]

        if current_char == "{":
            closing_index = cleaned_template.find("}", cursor + 1)
            if closing_index == -1:
                return None, [], "Column pattern template has an unmatched `{`."

            placeholder_name = cleaned_template[cursor + 1 : closing_index].strip()
            if not placeholder_name:
                return (
                    None,
                    [],
                    "Column pattern template contains an empty placeholder.",
                )
            if placeholder_name not in SUPPORTED_PATTERN_PLACEHOLDERS:
                return (
                    None,
                    [],
                    "Unsupported placeholder `{%s}`. Use `{scale}` and optionally `{rater}`."
                    % placeholder_name,
                )
            if placeholder_name in placeholders:
                return (
                    None,
                    [],
                    "Each placeholder can appear only once in the column pattern template.",
                )
            if previous_was_placeholder:
                return (
                    None,
                    [],
                    "Adjacent placeholders need a literal separator in the column pattern template.",
                )

            placeholders.append(placeholder_name)
            regex_parts.append(f"(?P<{placeholder_name}>.+?)")
            cursor = closing_index + 1
            previous_was_placeholder = True
            continue

        if current_char == "}":
            return None, [], "Column pattern template has an unmatched `}`."

        literal_start = cursor
        while cursor < len(cleaned_template) and cleaned_template[cursor] not in "{}":
            cursor += 1
        regex_parts.append(re.escape(cleaned_template[literal_start:cursor]))
        previous_was_placeholder = False

    if "scale" not in placeholders:
        return None, [], "Column pattern template must include a `{scale}` placeholder."

    return re.compile("^" + "".join(regex_parts) + "$"), placeholders, None


def parse_annotation_column_pattern(
    annotation_columns: Sequence[str],
    template: str,
) -> ColumnPatternParseResult:
    pattern, placeholders, error = _compile_column_pattern_template(template)
    column_metadata: Dict[str, ColumnPatternMetadata] = {}
    matched_columns: List[str] = []
    unmatched_columns: List[str] = []
    preview_rows: List[Dict[str, str]] = []

    for column_name in annotation_columns:
        if error:
            unmatched_columns.append(column_name)
            preview_rows.append(
                {
                    "Annotation column": column_name,
                    "Matched": "No",
                    "Scale": "",
                    "Rater": "",
                    "Notes": error,
                }
            )
            continue

        assert pattern is not None
        match = pattern.match(column_name)
        if match is None:
            unmatched_columns.append(column_name)
            preview_rows.append(
                {
                    "Annotation column": column_name,
                    "Matched": "No",
                    "Scale": "",
                    "Rater": "",
                    "Notes": "Does not match the current template.",
                }
            )
            continue

        scale_name = match.groupdict().get("scale", "").strip()
        rater_name = match.groupdict().get("rater", "").strip()
        if not scale_name:
            unmatched_columns.append(column_name)
            preview_rows.append(
                {
                    "Annotation column": column_name,
                    "Matched": "No",
                    "Scale": "",
                    "Rater": "",
                    "Notes": "The `{scale}` capture is empty.",
                }
            )
            continue

        metadata: ColumnPatternMetadata = {"scale": scale_name}
        if "rater" in placeholders and rater_name:
            metadata["rater"] = rater_name

        column_metadata[column_name] = metadata
        matched_columns.append(column_name)
        preview_rows.append(
            {
                "Annotation column": column_name,
                "Matched": "Yes",
                "Scale": scale_name,
                "Rater": metadata.get("rater", ""),
                "Notes": "Matched",
            }
        )

    matched_scale_groups: Dict[str, List[str]] = {}
    for column_name, metadata in column_metadata.items():
        matched_scale_groups.setdefault(metadata["scale"], []).append(column_name)

    matched_scale_groups = {
        scale_name: sorted(columns)
        for scale_name, columns in sorted(matched_scale_groups.items())
    }
    skipped_scales = {
        scale_name: columns
        for scale_name, columns in matched_scale_groups.items()
        if len(columns) < 2
    }

    return {
        "template": template,
        "error": error,
        "placeholders": placeholders,
        "column_metadata": column_metadata,
        "matched_columns": sorted(matched_columns),
        "unmatched_columns": sorted(set(unmatched_columns)),
        "matched_scale_groups": matched_scale_groups,
        "skipped_scales": skipped_scales,
        "preview_rows": preview_rows,
        "supports_rater": "rater" in placeholders,
    }


def _resolve_column_metadata(
    mappings: Sequence[EvaluationMapping],
    column_metadata: Optional[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, ColumnPatternMetadata]:
    if column_metadata is not None:
        resolved_metadata: Dict[str, ColumnPatternMetadata] = {}
        for column_name, metadata in column_metadata.items():
            normalized_metadata: ColumnPatternMetadata = {}
            scale_name = str(metadata.get("scale", "")).strip()
            rater_name = str(metadata.get("rater", "")).strip()
            if scale_name:
                normalized_metadata["scale"] = scale_name
            if rater_name:
                normalized_metadata["rater"] = rater_name
            if normalized_metadata:
                resolved_metadata[str(column_name)] = normalized_metadata
        return resolved_metadata

    all_annotation_columns = sorted(
        {
            column
            for mapping in mappings
            for column in mapping.get("annotation_columns", mapping["human_columns"])
        }
    )
    return parse_annotation_column_pattern(
        all_annotation_columns,
        DEFAULT_COLUMN_PATTERN_TEMPLATE,
    )["column_metadata"]


def _display_rater_name(rater_key: str) -> str:
    return rater_key


def _sorted_unique_values(values: Sequence[Any]) -> List[Any]:
    unique_values = list(dict.fromkeys(value for value in values if pd.notna(value)))
    try:
        return sorted(unique_values)
    except TypeError:
        return sorted(unique_values, key=str)


def _extract_numeric_value(value: Any, target_type: str) -> Any:
    normalized = _normalize_missing_value(value)
    if normalized is pd.NA:
        return pd.NA

    if isinstance(normalized, bool):
        normalized = int(normalized)

    if isinstance(normalized, (int, float)):
        if target_type == "Integer":
            return int(normalized)
        return float(normalized)

    text_value = str(normalized).replace(",", ".")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text_value)
    if not match:
        return pd.NA

    numeric_value = float(match.group(0))
    if target_type == "Integer":
        return int(numeric_value)
    return numeric_value


def _coerce_annotation_series(series: pd.Series, label_type: str) -> pd.Series:
    normalized = series.map(_normalize_missing_value)
    if label_type == "Text":
        return normalized.map(
            lambda value: pd.NA if value is pd.NA else str(value)
        ).astype("string")

    if label_type == "Integer":
        return pd.Series(
            normalized.map(lambda value: _extract_numeric_value(value, "Integer")),
            index=series.index,
            dtype="Int64",
        )

    return pd.to_numeric(
        pd.Series(
            normalized.map(lambda value: _extract_numeric_value(value, "Float")),
            index=series.index,
        ),
        errors="coerce",
    )


def _derived_prediction_column_name(mapping_id: str) -> str:
    return f"__metrics_only_prediction_{mapping_id}"


def _derived_comparison_description(label_type: str) -> str:
    if label_type == "Float":
        return "mean across selected annotation columns"
    return "majority vote across selected annotation columns"


def _build_derived_prediction_series(
    data: pd.DataFrame,
    annotation_columns: Sequence[str],
    label_type: str,
) -> pd.Series:
    valid_annotation_columns = [
        column for column in annotation_columns if column in data.columns
    ]
    if not valid_annotation_columns:
        return pd.Series([pd.NA] * len(data), index=data.index, dtype="object")

    coerced_df = pd.DataFrame(
        {
            column: _coerce_annotation_series(data[column], label_type)
            for column in valid_annotation_columns
        },
        index=data.index,
    )

    if label_type == "Float":
        return coerced_df.mean(axis=1, skipna=True)

    majority_vote = compute_majority_vote(
        {column: coerced_df[column].tolist() for column in valid_annotation_columns}
    )
    if label_type == "Integer":
        return pd.Series(majority_vote, index=data.index, dtype="Int64")
    return pd.Series(majority_vote, index=data.index, dtype="string")


def build_metrics_only_results_df(
    data: pd.DataFrame,
    mappings: Sequence[EvaluationMapping],
) -> pd.DataFrame:
    results_df = data.copy()
    for mapping in mappings:
        results_df[mapping["llm_field"]] = _build_derived_prediction_series(
            results_df,
            mapping.get("annotation_columns", mapping["human_columns"]),
            mapping["label_type"],
        )
    return results_df


def compute_scale_correlations_by_rater(
    data: pd.DataFrame,
    mappings: Sequence[EvaluationMapping],
    *,
    column_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    method: str = "spearman",
    min_overlap: int = 3,
) -> Dict[str, Any]:
    numeric_mappings = [
        mapping for mapping in mappings if mapping["label_type"] in {"Integer", "Float"}
    ]
    skipped_non_numeric = [
        mapping["name"]
        for mapping in mappings
        if mapping["label_type"] not in {"Integer", "Float"}
    ]

    rater_scale_values: Dict[str, Dict[str, pd.Series]] = {}
    unmatched_columns: List[str] = []
    duplicate_assignments: List[str] = []
    resolved_column_metadata = _resolve_column_metadata(mappings, column_metadata)

    for mapping in numeric_mappings:
        scale_name = mapping["name"]
        for column in mapping.get("annotation_columns", mapping["human_columns"]):
            if column not in data.columns:
                continue

            metadata = resolved_column_metadata.get(column, {})
            rater_key = metadata.get("rater")
            if not rater_key:
                unmatched_columns.append(column)
                continue

            scale_values = rater_scale_values.setdefault(rater_key, {})
            if scale_name in scale_values:
                duplicate_assignments.append(f"{rater_key}:{scale_name}")
                continue

            scale_values[scale_name] = _coerce_annotation_series(
                data[column], mapping["label_type"]
            )

    by_rater: Dict[str, Dict[str, Any]] = {}
    skipped_raters: Dict[str, int] = {}
    aggregated_pair_rows: List[Dict[str, Any]] = []

    for rater_key, scale_series in sorted(rater_scale_values.items()):
        if len(scale_series) < 2:
            skipped_raters[rater_key] = len(scale_series)
            continue

        scale_frame = pd.DataFrame(scale_series, index=data.index)
        scale_names = sorted(scale_frame.columns.tolist())
        correlation_matrix = pd.DataFrame(
            index=scale_names,
            columns=scale_names,
            dtype=float,
        )
        overlap_matrix = pd.DataFrame(
            index=scale_names,
            columns=scale_names,
            dtype="Int64",
        )
        pairwise_rows: List[Dict[str, Any]] = []

        for left_scale in scale_names:
            for right_scale in scale_names:
                pair_data = scale_frame[[left_scale, right_scale]].dropna()
                overlap = len(pair_data)
                overlap_matrix.loc[left_scale, right_scale] = overlap

                if left_scale == right_scale:
                    correlation_matrix.loc[left_scale, right_scale] = (
                        1.0 if overlap > 0 else np.nan
                    )
                    continue

                if overlap < min_overlap:
                    correlation_matrix.loc[left_scale, right_scale] = np.nan
                    continue

                correlation = pair_data[left_scale].corr(
                    pair_data[right_scale],
                    method=method,
                )
                correlation_matrix.loc[left_scale, right_scale] = correlation

                if left_scale < right_scale:
                    pairwise_row = {
                        "Scale A": left_scale,
                        "Scale B": right_scale,
                        "Correlation": correlation,
                        "Absolute correlation": (
                            abs(float(correlation)) if pd.notna(correlation) else np.nan
                        ),
                        "Overlap": overlap,
                    }
                    pairwise_rows.append(pairwise_row)
                    aggregated_pair_rows.append(
                        {
                            **pairwise_row,
                            "Rater": _display_rater_name(rater_key),
                            "_rater_key": rater_key,
                        }
                    )

        pairwise_df = pd.DataFrame(pairwise_rows)
        if not pairwise_df.empty:
            pairwise_df = pairwise_df.sort_values(
                by=["Absolute correlation", "Overlap", "Scale A", "Scale B"],
                ascending=[False, False, True, True],
            ).reset_index(drop=True)

        by_rater[rater_key] = {
            "scale_frame": scale_frame,
            "correlation_matrix": correlation_matrix,
            "overlap_matrix": overlap_matrix,
            "pairwise_df": pairwise_df,
            "n_scales": len(scale_names),
        }

    summary_rows = []
    for rater_key, result in by_rater.items():
        pairwise_df = result["pairwise_df"]
        strongest_pair = "N/A"
        strongest_corr = np.nan
        strongest_overlap = np.nan
        if not pairwise_df.empty:
            top_row = pairwise_df.iloc[0]
            strongest_pair = f"{top_row['Scale A']} vs {top_row['Scale B']}"
            strongest_corr = top_row["Correlation"]
            strongest_overlap = top_row["Overlap"]

        summary_rows.append(
            {
                "Rater": _display_rater_name(rater_key),
                "Scales": result["n_scales"],
                "Pairwise comparisons": len(pairwise_df),
                "Mean |correlation|": (
                    float(pairwise_df["Absolute correlation"].mean())
                    if not pairwise_df.empty
                    else np.nan
                ),
                "Max |correlation|": (
                    float(pairwise_df["Absolute correlation"].max())
                    if not pairwise_df.empty
                    else np.nan
                ),
                "Strongest pair": strongest_pair,
                "Overlap": strongest_overlap,
                "_rater_key": rater_key,
                "_strongest_correlation": strongest_corr,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["Max |correlation|", "Scales", "Rater"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    aggregated_pairwise_df = pd.DataFrame(aggregated_pair_rows)
    strongest_pairs_overall_df = pd.DataFrame()
    if not aggregated_pairwise_df.empty:
        strongest_pairs_overall_df = (
            aggregated_pairwise_df.groupby(["Scale A", "Scale B"], dropna=False)
            .agg(
                **{
                    "Average correlation": ("Correlation", "mean"),
                    "Average |correlation|": ("Absolute correlation", "mean"),
                    "Correlation std": ("Correlation", "std"),
                    "Raters": ("_rater_key", "nunique"),
                    "Average overlap": ("Overlap", "mean"),
                    "Min overlap": ("Overlap", "min"),
                    "Max overlap": ("Overlap", "max"),
                }
            )
            .reset_index()
            .sort_values(
                by=[
                    "Average |correlation|",
                    "Raters",
                    "Average overlap",
                    "Scale A",
                    "Scale B",
                ],
                ascending=[False, False, False, True, True],
            )
            .reset_index(drop=True)
        )

    return {
        "method": method,
        "min_overlap": min_overlap,
        "summary_df": summary_df,
        "by_rater": by_rater,
        "aggregated_pairwise_df": aggregated_pairwise_df,
        "strongest_pairs_overall_df": strongest_pairs_overall_df,
        "skipped_non_numeric": skipped_non_numeric,
        "unmatched_columns": sorted(set(unmatched_columns)),
        "duplicate_assignments": sorted(set(duplicate_assignments)),
        "skipped_raters": skipped_raters,
    }


def compute_annotator_agreement_across_scales(
    data: pd.DataFrame,
    mappings: Sequence[EvaluationMapping],
    *,
    column_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    min_overlap: int = 2,
) -> Dict[str, Any]:
    supported_mappings = [
        mapping for mapping in mappings if mapping["label_type"] in {"Integer", "Text"}
    ]
    skipped_mappings = [
        mapping["name"]
        for mapping in mappings
        if mapping["label_type"] not in {"Integer", "Text"}
    ]

    per_scale_rows: List[Dict[str, Any]] = []
    unmatched_columns: List[str] = []
    resolved_column_metadata = _resolve_column_metadata(mappings, column_metadata)

    for mapping in supported_mappings:
        scale_name = mapping["name"]
        label_type = mapping["label_type"]
        weights = mapping.get("kappa_weights") if label_type == "Integer" else None

        rater_columns: List[tuple[str, str]] = []
        for column in mapping.get("annotation_columns", mapping["human_columns"]):
            if column not in data.columns:
                continue

            metadata = resolved_column_metadata.get(column, {})
            rater_key = metadata.get("rater")
            if not rater_key:
                unmatched_columns.append(column)
                continue

            rater_columns.append((rater_key, column))

        rater_columns = sorted(rater_columns, key=lambda item: item[0])
        for index, (left_rater, left_column) in enumerate(rater_columns):
            left_series = _coerce_annotation_series(data[left_column], label_type)
            for right_rater, right_column in rater_columns[index + 1 :]:
                right_series = _coerce_annotation_series(data[right_column], label_type)
                pair_df = pd.DataFrame(
                    {
                        "left": left_series,
                        "right": right_series,
                    },
                    index=data.index,
                ).dropna()
                overlap = len(pair_df)
                if overlap < min_overlap:
                    continue

                disagreement_count = int((pair_df["left"] != pair_df["right"]).sum())
                conflict_percentage = (
                    disagreement_count / overlap if overlap > 0 else np.nan
                )

                labels = _sorted_unique_values(
                    pair_df["left"].tolist() + pair_df["right"].tolist()
                )
                agreement = compute_cohens_kappa(
                    pair_df["left"].tolist(),
                    pair_df["right"].tolist(),
                    labels=labels,
                    weights=weights,
                )
                per_scale_rows.append(
                    {
                        "Scale": scale_name,
                        "Annotator A": _display_rater_name(left_rater),
                        "Annotator B": _display_rater_name(right_rater),
                        "Agreement": agreement,
                        "Absolute agreement": (
                            abs(float(agreement)) if pd.notna(agreement) else np.nan
                        ),
                        "Overlap": overlap,
                        "Disagreements": disagreement_count,
                        "Conflict %": conflict_percentage,
                        "Label type": label_type,
                        "Correction": (
                            next(
                                (
                                    label
                                    for label, value in KAPPA_WEIGHT_OPTIONS.items()
                                    if value == weights
                                ),
                                "Unweighted",
                            )
                            if label_type == "Integer"
                            else "Unweighted"
                        ),
                    }
                )

    per_scale_df = pd.DataFrame(per_scale_rows)
    aggregated_pairs_df = pd.DataFrame()
    annotator_summary_df = pd.DataFrame()

    if not per_scale_df.empty:
        aggregated_pairs_df = (
            per_scale_df.groupby(["Annotator A", "Annotator B"], dropna=False)
            .agg(
                **{
                    "Average agreement": ("Agreement", "mean"),
                    "Average |agreement|": ("Absolute agreement", "mean"),
                    "Agreement std": ("Agreement", "std"),
                    "Scales": ("Scale", "nunique"),
                    "Average overlap": ("Overlap", "mean"),
                    "Min overlap": ("Overlap", "min"),
                    "Max overlap": ("Overlap", "max"),
                }
            )
            .reset_index()
            .sort_values(
                by=[
                    "Average agreement",
                    "Scales",
                    "Average overlap",
                    "Annotator A",
                    "Annotator B",
                ],
                ascending=[False, False, False, True, True],
            )
            .reset_index(drop=True)
        )

        annotators = sorted(
            set(aggregated_pairs_df["Annotator A"]).union(
                aggregated_pairs_df["Annotator B"]
            )
        )
        summary_rows = []
        for annotator in annotators:
            annotator_pairs = aggregated_pairs_df[
                (aggregated_pairs_df["Annotator A"] == annotator)
                | (aggregated_pairs_df["Annotator B"] == annotator)
            ].copy()
            if annotator_pairs.empty:
                continue

            annotator_pairs["Peer"] = annotator_pairs.apply(
                lambda row: (
                    row["Annotator B"]
                    if row["Annotator A"] == annotator
                    else row["Annotator A"]
                ),
                axis=1,
            )
            best_peer_row = annotator_pairs.sort_values(
                by=["Average agreement", "Scales", "Average overlap", "Peer"],
                ascending=[False, False, False, True],
            ).iloc[0]

            summary_rows.append(
                {
                    "Annotator": annotator,
                    "Average agreement to others": float(
                        annotator_pairs["Average agreement"].mean()
                    ),
                    "Average |agreement| to others": float(
                        annotator_pairs["Average |agreement|"].mean()
                    ),
                    "Peers compared": len(annotator_pairs),
                    "Scales covered": int(annotator_pairs["Scales"].sum()),
                    "Closest peer": best_peer_row["Peer"],
                    "Best average agreement": float(best_peer_row["Average agreement"]),
                }
            )

        annotator_summary_df = pd.DataFrame(summary_rows)
        if not annotator_summary_df.empty:
            annotator_summary_df = annotator_summary_df.sort_values(
                by=[
                    "Average agreement to others",
                    "Average |agreement| to others",
                    "Annotator",
                ],
                ascending=[False, False, True],
            ).reset_index(drop=True)

    return {
        "per_scale_df": per_scale_df,
        "aggregated_pairs_df": aggregated_pairs_df,
        "annotator_summary_df": annotator_summary_df,
        "skipped_mappings": skipped_mappings,
        "unmatched_columns": sorted(set(unmatched_columns)),
        "min_overlap": min_overlap,
    }


def build_aggregated_correlation_matrix(
    strongest_pairs_overall_df: pd.DataFrame,
    *,
    value_column: str = "Average |correlation|",
) -> pd.DataFrame:
    if strongest_pairs_overall_df.empty:
        return pd.DataFrame()

    scale_names = sorted(
        set(strongest_pairs_overall_df["Scale A"]).union(
            strongest_pairs_overall_df["Scale B"]
        )
    )
    matrix = pd.DataFrame(
        np.nan,
        index=scale_names,
        columns=scale_names,
        dtype=float,
    )

    diagonal_value = 1.0 if value_column == "Average |correlation|" else 1.0
    for scale_name in scale_names:
        matrix.loc[scale_name, scale_name] = diagonal_value

    for _, row in strongest_pairs_overall_df.iterrows():
        left_scale = row["Scale A"]
        right_scale = row["Scale B"]
        matrix.loc[left_scale, right_scale] = row[value_column]
        matrix.loc[right_scale, left_scale] = row[value_column]

    return matrix


def build_lower_triangle_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return matrix.copy()

    lower_triangle = matrix.copy().astype(float)
    upper_triangle_indices = np.triu_indices_from(lower_triangle, k=1)
    lower_triangle.values[upper_triangle_indices] = np.nan
    return lower_triangle


def build_pair_metric_matrix(
    pairs_df: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    value_col: str,
    diagonal_value: float = 1.0,
) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame()

    labels = sorted(set(pairs_df[left_col]).union(pairs_df[right_col]))
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    for label in labels:
        matrix.loc[label, label] = diagonal_value

    for _, row in pairs_df.iterrows():
        left_value = row[left_col]
        right_value = row[right_col]
        metric_value = row[value_col]
        matrix.loc[left_value, right_value] = metric_value
        matrix.loc[right_value, left_value] = metric_value

    return matrix


def _render_triangular_correlation_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    value_label: str,
    absolute: bool,
) -> None:
    triangular_matrix = build_lower_triangle_matrix(matrix)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig_width = max(6, min(1.2 * len(triangular_matrix.columns), 16))
        fig_height = max(5, min(1.0 * len(triangular_matrix.index), 14))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        mask = triangular_matrix.isna()

        if absolute:
            cmap = sns.color_palette("YlOrRd", as_cmap=True)
            sns.heatmap(
                triangular_matrix,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                linewidths=0.5,
                linecolor="white",
                square=True,
                mask=mask,
                cbar_kws={"label": value_label},
                ax=ax,
            )
        else:
            sns.heatmap(
                triangular_matrix,
                annot=True,
                fmt=".2f",
                cmap="vlag",
                center=0.0,
                vmin=-1.0,
                vmax=1.0,
                linewidths=0.5,
                linecolor="white",
                square=True,
                mask=mask,
                cbar_kws={"label": value_label},
                ax=ax,
            )
        ax.set_title(title)
        ax.set_xlabel("Scale")
        ax.set_ylabel("Scale")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except ModuleNotFoundError:
        st.info(
            "Heatmap rendering is unavailable because `matplotlib` or `seaborn` is not installed in this environment."
        )
        st.dataframe(triangular_matrix.round(4), use_container_width=True)


def _build_metrics_only_mapping(
    data: pd.DataFrame,
    name: str,
    annotation_columns: Sequence[str],
    existing_mapping: Optional[Mapping[str, Any]] = None,
) -> EvaluationMapping:
    valid_annotation_columns = [
        column for column in annotation_columns if column in data.columns
    ]
    label_type = (
        str(existing_mapping.get("label_type"))
        if existing_mapping and existing_mapping.get("label_type") in LABEL_TYPE_OPTIONS
        else infer_label_type_for_columns(data, valid_annotation_columns)
    )
    mapping_id = (
        str(existing_mapping.get("id"))
        if existing_mapping and existing_mapping.get("id")
        else create_mapping_id()
    )
    kappa_weights = (
        existing_mapping.get("kappa_weights")
        if existing_mapping
        and existing_mapping.get("kappa_weights") in set(KAPPA_WEIGHT_OPTIONS.values())
        else None
    )
    raw_kripp_level = (
        str(existing_mapping.get("kripp_level_of_measurement"))
        if existing_mapping and existing_mapping.get("kripp_level_of_measurement")
        else None
    )
    kripp_level = (
        raw_kripp_level
        if raw_kripp_level in KRIPP_LEVEL_OPTIONS
        else default_kripp_level_for_label_type(label_type)
    )
    if label_type == "Text":
        kripp_level = "nominal"

    return {
        "id": mapping_id,
        "name": name,
        "llm_field": _derived_prediction_column_name(mapping_id),
        "human_columns": valid_annotation_columns,
        "annotation_columns": valid_annotation_columns,
        "label_type": label_type,
        "kappa_weights": kappa_weights,
        "kripp_level_of_measurement": kripp_level,
    }


def autodetect_scale_mappings(
    data: pd.DataFrame,
    annotation_columns: Sequence[str],
    *,
    existing_mappings: Optional[Sequence[Mapping[str, Any]]] = None,
    column_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[EvaluationMapping]:
    """
    Auto-group matched annotation columns into one mapping per extracted scale.
    """
    existing_by_name: Dict[str, Mapping[str, Any]] = {
        str(mapping.get("name")): mapping for mapping in (existing_mappings or [])
    }
    grouped_columns: Dict[str, List[str]] = {}
    resolved_column_metadata = column_metadata
    if resolved_column_metadata is None:
        resolved_column_metadata = parse_annotation_column_pattern(
            annotation_columns,
            DEFAULT_COLUMN_PATTERN_TEMPLATE,
        )["column_metadata"]

    for column in annotation_columns:
        metadata = resolved_column_metadata.get(column, {})
        scale_name = str(metadata.get("scale", "")).strip()
        if not scale_name:
            continue
        grouped_columns.setdefault(scale_name, []).append(column)

    detected_mappings: List[EvaluationMapping] = []
    for scale_name in sorted(grouped_columns):
        columns_for_scale = sorted(grouped_columns[scale_name])
        if len(columns_for_scale) < 2:
            continue
        detected_mappings.append(
            _build_metrics_only_mapping(
                data,
                name=scale_name,
                annotation_columns=columns_for_scale,
                existing_mapping=existing_by_name.get(scale_name),
            )
        )

    return detected_mappings


def _mapped_columns_with_rater_metadata(
    mappings: Sequence[EvaluationMapping],
    column_metadata: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    mapped_columns: List[str] = []
    for mapping in mappings:
        for column in mapping.get("annotation_columns", mapping["human_columns"]):
            if str(column_metadata.get(column, {}).get("rater", "")).strip():
                mapped_columns.append(column)
    return sorted(set(mapped_columns))


def _rater_analysis_unavailability_reason(
    mappings: Sequence[EvaluationMapping],
    parse_result: ColumnPatternParseResult,
) -> Optional[str]:
    if parse_result["error"]:
        return "Rater-aware analyses are unavailable until the column pattern template is valid."

    if not parse_result["supports_rater"]:
        return (
            "Rater-aware analyses are unavailable because the current column pattern "
            "template does not include `{rater}`."
        )

    if (
        len(
            _mapped_columns_with_rater_metadata(
                mappings, parse_result["column_metadata"]
            )
        )
        < 2
    ):
        return (
            "Rater-aware analyses are unavailable because fewer than two mapped "
            "annotation columns currently expose rater information from the template."
        )

    return None


def _render_metrics_only_mapping_editor(
    data: pd.DataFrame,
    available_annotation_columns: Sequence[str],
    column_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    autodetect_disabled_reason: Optional[str] = None,
) -> List[EvaluationMapping]:
    mapping_state_key = _state_key("evaluation_mappings")
    initialized_state_key = _state_key("evaluation_mappings_initialized")

    if not st.session_state.get(initialized_state_key, False):
        st.session_state[mapping_state_key] = autodetect_scale_mappings(
            data,
            available_annotation_columns,
            column_metadata=column_metadata,
        )
        st.session_state[initialized_state_key] = True

    if st.button(
        "Auto-detect mappings from the current pattern",
        key=_state_key("autodetect_mappings_button"),
        disabled=bool(autodetect_disabled_reason),
        help=autodetect_disabled_reason,
    ):
        st.session_state[mapping_state_key] = autodetect_scale_mappings(
            data,
            available_annotation_columns,
            existing_mappings=st.session_state.get(mapping_state_key, []),
            column_metadata=column_metadata,
        )
        _clear_metrics_only_cached_results()
        st.rerun()

    current_mappings = list(st.session_state.get(mapping_state_key, []))
    if not current_mappings:
        st.info(
            "No mappings were auto-detected from the current pattern yet. Adjust the template above or add mappings manually."
        )
    updated_mappings: List[EvaluationMapping] = []
    removed_mapping_id = None

    for index, mapping in enumerate(current_mappings):
        mapping_id = mapping["id"]
        st.markdown(f"**Mapping {index + 1}**")
        col1, col2, col3, col4, col5, col6 = st.columns(
            [1.2, 2.2, 0.95, 0.95, 1.15, 0.55]
        )

        name_key = _state_key(f"mapping_name_{mapping_id}")
        columns_key = _state_key(f"mapping_columns_{mapping_id}")
        type_key = _state_key(f"mapping_type_{mapping_id}")
        weights_key = _state_key(f"mapping_weights_{mapping_id}")
        kripp_key = _state_key(f"mapping_kripp_level_{mapping_id}")

        if name_key not in st.session_state:
            st.session_state[name_key] = mapping["name"]
        if columns_key not in st.session_state:
            st.session_state[columns_key] = mapping.get(
                "annotation_columns", mapping["human_columns"]
            )
        else:
            st.session_state[columns_key] = [
                column
                for column in st.session_state[columns_key]
                if column in available_annotation_columns
            ]
        if type_key not in st.session_state:
            st.session_state[type_key] = mapping["label_type"]
        if weights_key not in st.session_state:
            stored_weights = mapping.get("kappa_weights")
            st.session_state[weights_key] = next(
                (
                    label
                    for label, value in KAPPA_WEIGHT_OPTIONS.items()
                    if value == stored_weights
                ),
                "Unweighted",
            )
        if kripp_key not in st.session_state:
            stored_kripp_level = mapping.get("kripp_level_of_measurement")
            st.session_state[kripp_key] = (
                stored_kripp_level
                if stored_kripp_level in KRIPP_LEVEL_OPTIONS
                else default_kripp_level_for_label_type(mapping["label_type"])
            )

        with col1:
            mapping_name = st.text_input("Display name", key=name_key)

        with col2:
            annotation_columns = st.multiselect(
                "Annotation columns",
                options=list(available_annotation_columns),
                key=columns_key,
            )

        with col3:
            label_type = st.selectbox(
                "Label type",
                options=LABEL_TYPE_OPTIONS,
                key=type_key,
            )

        with col4:
            kappa_weights_label = st.selectbox(
                "Correction",
                options=list(KAPPA_WEIGHT_OPTIONS.keys()),
                key=weights_key,
            )

        with col5:
            if label_type == "Text":
                st.session_state[kripp_key] = "nominal"
            kripp_level = st.selectbox(
                "Krippendorff level",
                options=KRIPP_LEVEL_OPTIONS,
                key=kripp_key,
                disabled=label_type == "Text",
            )

        with col6:
            remove_clicked = st.button(
                "Remove",
                key=_state_key(f"mapping_remove_{mapping_id}"),
            )

        if remove_clicked:
            removed_mapping_id = mapping_id
        else:
            rebuilt_mapping = _build_metrics_only_mapping(
                data,
                name=mapping_name.strip() or mapping["name"],
                annotation_columns=annotation_columns,
                existing_mapping={
                    **mapping,
                    "label_type": label_type,
                    "kappa_weights": KAPPA_WEIGHT_OPTIONS[kappa_weights_label],
                    "kripp_level_of_measurement": (
                        "nominal" if label_type == "Text" else kripp_level
                    ),
                },
            )
            rebuilt_mapping["label_type"] = label_type
            rebuilt_mapping["kappa_weights"] = KAPPA_WEIGHT_OPTIONS[kappa_weights_label]
            rebuilt_mapping["kripp_level_of_measurement"] = (
                "nominal" if label_type == "Text" else kripp_level
            )
            updated_mappings.append(rebuilt_mapping)

            if not annotation_columns:
                st.warning(
                    f"Mapping `{mapping_name.strip() or mapping['name']}` needs at least one annotation column."
                )
            elif len(annotation_columns) == 1:
                st.warning(
                    f"Mapping `{mapping_name.strip() or mapping['name']}` only has one annotation column, so agreement metrics will be limited."
                )
            elif len(annotation_columns) < 3:
                st.warning(
                    f"Mapping `{mapping_name.strip() or mapping['name']}` has fewer than 3 annotation columns, so Alt-Test and Krippendorff will be skipped."
                )
            else:
                st.caption(
                    f"Derived comparison for model-style metrics: {_derived_comparison_description(label_type)}."
                )

        st.markdown("---")

    if st.button("Add mapping", key=_state_key("add_mapping_button")):
        new_name = f"Mapping {len(current_mappings) + 1}"
        default_columns = list(
            available_annotation_columns[: min(3, len(available_annotation_columns))]
        )
        updated_mappings.append(
            _build_metrics_only_mapping(
                data,
                name=new_name,
                annotation_columns=default_columns,
            )
        )
        st.session_state[mapping_state_key] = updated_mappings
        st.session_state[initialized_state_key] = True
        _clear_metrics_only_cached_results()
        st.rerun()

    if removed_mapping_id is not None:
        st.session_state[mapping_state_key] = updated_mappings
        st.session_state[initialized_state_key] = True
        _clear_metrics_only_cached_results()
        st.rerun()

    st.session_state[mapping_state_key] = updated_mappings
    st.session_state[initialized_state_key] = True
    return updated_mappings


CORRELATION_BY_RATER_METHOD = "Correlation by Rater"


def _render_scale_correlation_section(
    data: pd.DataFrame,
    mappings: Sequence[EvaluationMapping],
    column_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    st.markdown(
        """
        Examine how strongly each rater's scales move together across items.
        High absolute correlations can suggest that two scales may be partially redundant.
        Only **Integer** and **Float** mappings are included here.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox(
            "Correlation method",
            options=["spearman", "pearson", "kendall"],
            index=0,
            key=_state_key("scale_correlation_method"),
            help="Spearman is often a good default for ordinal scales.",
        )
    with col2:
        min_overlap = int(
            st.number_input(
                "Minimum overlapping annotations per pair",
                min_value=2,
                value=3,
                step=1,
                key=_state_key("scale_correlation_min_overlap"),
            )
        )
    matrix_view = st.selectbox(
        "Matrix value",
        options=["Absolute correlation", "Signed correlation"],
        index=0,
        key=_state_key("scale_correlation_matrix_view"),
        help="Absolute correlation highlights redundancy strength; signed correlation keeps direction.",
    )
    use_absolute_matrix = matrix_view == "Absolute correlation"

    if st.button(
        "Compute Scale Correlations",
        key=_state_key("compute_scale_correlations_button"),
    ):
        st.session_state[_state_key("scale_correlation_results")] = (
            compute_scale_correlations_by_rater(
                data,
                mappings,
                column_metadata=column_metadata,
                method=method,
                min_overlap=min_overlap,
            )
        )

    results = st.session_state.get(_state_key("scale_correlation_results"))
    if not results:
        st.info("Compute correlations to inspect redundancy between scales by rater.")
        return

    if results["skipped_non_numeric"]:
        st.caption(
            "Skipped non-numeric mappings: "
            + ", ".join(f"`{name}`" for name in results["skipped_non_numeric"])
        )
    if results["unmatched_columns"]:
        st.caption(
            "Could not extract a rater from the current pattern for: "
            + ", ".join(f"`{name}`" for name in results["unmatched_columns"])
        )
    if results["duplicate_assignments"]:
        st.caption(
            "Duplicate rater/scale assignments were ignored for: "
            + ", ".join(f"`{name}`" for name in results["duplicate_assignments"])
        )

    summary_df = results["summary_df"].copy()
    if summary_df.empty:
        st.warning(
            "No per-rater scale correlations could be computed. At least one rater needs annotations for 2 numeric scales."
        )
        return

    display_summary = summary_df.drop(columns=["_rater_key", "_strongest_correlation"])
    display_summary["Mean |correlation|"] = display_summary["Mean |correlation|"].apply(
        lambda value: "N/A" if pd.isna(value) else f"{float(value):.4f}"
    )
    display_summary["Max |correlation|"] = display_summary["Max |correlation|"].apply(
        lambda value: "N/A" if pd.isna(value) else f"{float(value):.4f}"
    )
    st.subheader("Rater Summary")
    st.table(display_summary)

    for _, row in summary_df.iterrows():
        rater_key = row["_rater_key"]
        rater_result = results["by_rater"][rater_key]
        st.markdown(f"#### {_display_rater_name(rater_key)}")

        correlation_matrix = rater_result["correlation_matrix"].copy().astype(float)
        display_correlation_matrix = (
            correlation_matrix.abs() if use_absolute_matrix else correlation_matrix
        )
        overlap_matrix = rater_result["overlap_matrix"].copy()

        st.write("**Correlation matrix**")
        _render_triangular_correlation_heatmap(
            display_correlation_matrix,
            title=f"Scale Correlation Matrix for {_display_rater_name(rater_key)}",
            value_label=matrix_view,
            absolute=use_absolute_matrix,
        )

        st.write("**Pairwise overlap**")
        st.dataframe(overlap_matrix, use_container_width=True)

        pairwise_df = rater_result["pairwise_df"].copy()
        if pairwise_df.empty:
            st.info("No scale pairs met the minimum overlap threshold for this rater.")
            continue

        pairwise_display = pairwise_df.copy()
        pairwise_display["Correlation"] = pairwise_display["Correlation"].apply(
            lambda value: "N/A" if pd.isna(value) else f"{float(value):+.4f}"
        )
        pairwise_display["Absolute correlation"] = pairwise_display[
            "Absolute correlation"
        ].apply(lambda value: "N/A" if pd.isna(value) else f"{float(value):.4f}")
        st.write("**Strongest scale pairs**")
        st.dataframe(pairwise_display, use_container_width=True)

    strongest_pairs_overall_df = results["strongest_pairs_overall_df"].copy()
    st.subheader("Strongest Scale Pairs Across Raters")
    if strongest_pairs_overall_df.empty:
        st.info("No cross-rater scale pairs were available to aggregate.")
        return

    aggregate_view = st.selectbox(
        "Aggregate matrix value",
        options=["Average absolute correlation", "Average signed correlation"],
        index=0,
        key=_state_key("scale_correlation_aggregate_view"),
    )
    value_column = (
        "Average |correlation|"
        if aggregate_view == "Average absolute correlation"
        else "Average correlation"
    )
    aggregate_matrix = build_aggregated_correlation_matrix(
        strongest_pairs_overall_df,
        value_column=value_column,
    )

    _render_triangular_correlation_heatmap(
        aggregate_matrix,
        title="Scale Correlation Matrix Averaged Across Raters",
        value_label=value_column,
        absolute=aggregate_view == "Average absolute correlation",
    )

    if st.checkbox(
        "Show aggregated pair details",
        key=_state_key("show_aggregated_pair_details"),
    ):
        strongest_pairs_display = strongest_pairs_overall_df.copy()
        for column_name in [
            "Average correlation",
            "Average |correlation|",
            "Correlation std",
            "Average overlap",
        ]:
            strongest_pairs_display[column_name] = strongest_pairs_display[
                column_name
            ].apply(lambda value: "N/A" if pd.isna(value) else f"{float(value):.4f}")
        st.dataframe(strongest_pairs_display, use_container_width=True)


def _render_annotator_agreement_section(
    data: pd.DataFrame,
    mappings: Sequence[EvaluationMapping],
    column_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    st.markdown("---")
    st.subheader("Annotator Agreement Across Scales")
    st.markdown(
        """
        Compare annotators directly across the configured scales to see who tends to
        annotate most like the others. This view uses **pairwise Cohen's kappa**
        within each scale, then averages those scores across scales.
        Only **Integer** and **Text** mappings are included here.
        """
    )

    min_overlap = int(
        st.number_input(
            "Minimum overlapping annotations per annotator pair",
            min_value=2,
            value=2,
            step=1,
            key=_state_key("annotator_agreement_min_overlap"),
        )
    )
    matrix_view = st.selectbox(
        "Annotator matrix value",
        options=["Average agreement", "Average absolute agreement"],
        index=0,
        key=_state_key("annotator_agreement_matrix_view"),
    )
    use_absolute_matrix = matrix_view == "Average absolute agreement"

    if st.button(
        "Compute Annotator Agreement",
        key=_state_key("compute_annotator_agreement_button"),
    ):
        st.session_state[_state_key("annotator_agreement_results")] = (
            compute_annotator_agreement_across_scales(
                data,
                mappings,
                column_metadata=column_metadata,
                min_overlap=min_overlap,
            )
        )

    results = st.session_state.get(_state_key("annotator_agreement_results"))
    if not results:
        st.info(
            "Compute annotator agreement to compare who agrees most with the others."
        )
        return

    if results["skipped_mappings"]:
        st.caption(
            "Skipped mappings that are not suitable for pairwise kappa: "
            + ", ".join(f"`{name}`" for name in results["skipped_mappings"])
        )
    if results["unmatched_columns"]:
        st.caption(
            "Could not extract an annotator from the current pattern for: "
            + ", ".join(f"`{name}`" for name in results["unmatched_columns"])
        )

    summary_df = results["annotator_summary_df"].copy()
    if summary_df.empty:
        st.warning(
            "No annotator agreement scores could be computed. At least two annotators need overlapping annotations on the same supported scales."
        )
        return

    display_summary = summary_df.copy()
    for column_name in [
        "Average agreement to others",
        "Average |agreement| to others",
        "Best average agreement",
    ]:
        display_summary[column_name] = display_summary[column_name].apply(
            lambda value: "N/A" if pd.isna(value) else f"{float(value):+.4f}"
        )
    st.write("**Annotator Summary**")
    st.table(display_summary)

    aggregated_pairs_df = results["aggregated_pairs_df"].copy()
    value_column = "Average |agreement|" if use_absolute_matrix else "Average agreement"
    annotator_matrix = build_pair_metric_matrix(
        aggregated_pairs_df,
        left_col="Annotator A",
        right_col="Annotator B",
        value_col=value_column,
    )
    st.write("**Annotator Agreement Matrix**")
    _render_triangular_correlation_heatmap(
        annotator_matrix,
        title="Annotator Agreement Matrix Averaged Across Scales",
        value_label=matrix_view,
        absolute=use_absolute_matrix,
    )

    if st.checkbox(
        "Show annotator pair details",
        key=_state_key("show_annotator_agreement_details"),
    ):
        display_pairs = aggregated_pairs_df.copy()
        for column_name in [
            "Average agreement",
            "Average |agreement|",
            "Agreement std",
            "Average overlap",
        ]:
            display_pairs[column_name] = display_pairs[column_name].apply(
                lambda value: "N/A" if pd.isna(value) else f"{float(value):+.4f}"
            )
        st.dataframe(display_pairs, use_container_width=True)

    if st.checkbox(
        "Show per-scale annotator agreement",
        key=_state_key("show_per_scale_annotator_agreement"),
    ):
        per_scale_df = results["per_scale_df"].copy()
        per_scale_df["Agreement"] = per_scale_df["Agreement"].apply(
            lambda value: "N/A" if pd.isna(value) else f"{float(value):+.4f}"
        )
        per_scale_df["Absolute agreement"] = per_scale_df["Absolute agreement"].apply(
            lambda value: "N/A" if pd.isna(value) else f"{float(value):.4f}"
        )
        per_scale_df["Conflict %"] = per_scale_df["Conflict %"].apply(
            lambda value: "N/A" if pd.isna(value) else f"{float(value):.1%}"
        )
        st.dataframe(per_scale_df, use_container_width=True)


def run_metrics_only_page() -> None:
    """
    Render the metrics-only page.
    """
    st.title("Metrics Only")
    st.markdown(
        """
        Compute metrics directly from existing annotation columns, without running the
        LLM annotation workflow first. Use a simple column pattern template to help the
        page auto-detect scales and, when available, rater identities.
        """
    )

    data = _upload_metrics_dataset()
    if data is None:
        st.info("Upload a dataset to configure mappings and compute metrics.")
        return

    st.markdown("### Step 2: Choose Annotation Columns", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 2", expanded=True):
        selection_key = _state_key("annotation_columns_selection")
        pattern_key = _state_key("column_pattern_template")
        applied_pattern_key = _state_key("applied_column_pattern_template")
        mapping_state_key = _state_key("evaluation_mappings")
        initialized_state_key = _state_key("evaluation_mappings_initialized")
        if pattern_key not in st.session_state:
            st.session_state[pattern_key] = DEFAULT_COLUMN_PATTERN_TEMPLATE
        column_pattern_template = st.text_input(
            "Column pattern template",
            key=pattern_key,
            help=(
                "Use `{scale}` and optionally `{rater}`. Examples: "
                "`Rater_{rater}_{scale}`, `{scale}_{rater}`, or `judge-{rater}-{scale}`."
            ),
        )

        previous_columns = st.session_state.get(_state_key("annotation_columns"), [])
        valid_previous_columns = [
            column for column in previous_columns if column in data.columns
        ]
        current_selection_state = st.session_state.get(
            selection_key, valid_previous_columns
        )
        st.session_state[selection_key] = [
            column for column in current_selection_state if column in data.columns
        ]

        dataset_parse_result = parse_annotation_column_pattern(
            data.columns.tolist(),
            column_pattern_template,
        )
        previous_template = st.session_state.get(applied_pattern_key)
        mappings_initialized = st.session_state.get(initialized_state_key, False)

        should_sync_selection_from_pattern = (
            column_pattern_template != previous_template
            or (not mappings_initialized and not st.session_state[selection_key])
        )
        if should_sync_selection_from_pattern:
            if dataset_parse_result["error"] is None:
                matched_columns = dataset_parse_result["matched_columns"]
                st.session_state[selection_key] = list(matched_columns)
                st.session_state[_state_key("annotation_columns")] = list(
                    matched_columns
                )
                st.session_state[mapping_state_key] = autodetect_scale_mappings(
                    data,
                    matched_columns,
                    existing_mappings=st.session_state.get(mapping_state_key, []),
                    column_metadata=dataset_parse_result["column_metadata"],
                )
            elif not st.session_state.get(mapping_state_key):
                st.session_state[mapping_state_key] = []
            st.session_state[initialized_state_key] = True
            st.session_state[applied_pattern_key] = column_pattern_template
            _clear_metrics_only_cached_results()
            st.rerun()

        annotation_columns = st.multiselect(
            "Annotation column pool:",
            options=data.columns.tolist(),
            key=selection_key,
            help=(
                "Columns matching the current template are preselected automatically. "
                "You can still adjust the selection manually if needed."
            ),
        )
        st.session_state[_state_key("annotation_columns")] = annotation_columns
        parse_result = parse_annotation_column_pattern(
            annotation_columns,
            column_pattern_template,
        )

        if dataset_parse_result["error"]:
            st.error(dataset_parse_result["error"])
        else:
            st.caption(
                f"The current template matches {len(dataset_parse_result['matched_columns'])} dataset columns across "
                f"{len(dataset_parse_result['matched_scale_groups'])} detected scale group(s). Matching columns are preselected below."
            )

        if not dataset_parse_result["supports_rater"]:
            st.caption(
                "This template does not capture `{rater}`, so rater-aware analyses will stay hidden."
            )
        if parse_result["unmatched_columns"]:
            st.warning(
                "These selected annotation columns do not match the current template: "
                + ", ".join(
                    f"`{column}`" for column in parse_result["unmatched_columns"]
                )
            )
        if dataset_parse_result["skipped_scales"]:
            st.warning(
                "These detected scales currently have fewer than 2 matching columns and will not be auto-created: "
                + ", ".join(
                    f"`{scale}` ({len(columns)} column)"
                    + ("" if len(columns) == 1 else "s")
                    for scale, columns in dataset_parse_result["skipped_scales"].items()
                )
            )
        if annotation_columns:
            st.write("Pattern preview")
            st.dataframe(
                pd.DataFrame(parse_result["preview_rows"]),
                use_container_width=True,
                hide_index=True,
            )

        if annotation_columns != previous_columns:
            if parse_result["error"] is None:
                st.session_state[mapping_state_key] = autodetect_scale_mappings(
                    data,
                    annotation_columns,
                    existing_mappings=st.session_state.get(mapping_state_key, []),
                    column_metadata=parse_result["column_metadata"],
                )
            elif not st.session_state.get(mapping_state_key):
                st.session_state[mapping_state_key] = []
            st.session_state[initialized_state_key] = True
            _clear_metrics_only_cached_results()
            st.rerun()

    if not annotation_columns:
        st.info("Select at least one annotation column to define scale mappings.")
        return

    st.markdown("### Step 3: Configure Scale Mappings", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 3", expanded=True):
        st.markdown(
            """
            Mappings are auto-detected from annotation columns that match the current pattern template.
            You can still edit everything manually after auto-detection.
            The page does not ask you to choose a comparison column manually. Instead, it derives
            one from the selected annotation columns for each mapping:
            - **Integer** and **Text** mappings use the majority vote.
            - **Float** mappings use the mean value.
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Alt-Test and Krippendorff require at least 3 annotation columns in a mapping."
        )
        mappings = _render_metrics_only_mapping_editor(
            data,
            annotation_columns,
            column_metadata=parse_result["column_metadata"],
            autodetect_disabled_reason=parse_result["error"],
        )

    st.markdown("### Step 4: Compute Metrics", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 4", expanded=True):
        rater_unavailability_reason = _rater_analysis_unavailability_reason(
            mappings,
            parse_result,
        )
        if rater_unavailability_reason:
            st.caption(rater_unavailability_reason)

        metrics_results_df = build_metrics_only_results_df(data, mappings)
        render_evaluation_dashboard(
            results_df=metrics_results_df,
            evaluation_mappings=mappings,
            ui_key_prefix=METRICS_ONLY_PREFIX,
            comparison_label="Derived comparison",
            comparison_entity_name="Consensus",
            annotation_column_label="Annotation columns",
            show_comparison_column=False,
            use_mapping_kappa_weights=True,
            use_mapping_kripp_level_of_measurement=True,
            enable_classical_krippendorff=True,
            custom_method_renderers=(
                {
                    CORRELATION_BY_RATER_METHOD: lambda: _render_scale_correlation_section(
                        data,
                        mappings,
                        column_metadata=parse_result["column_metadata"],
                    )
                }
                if rater_unavailability_reason is None
                else None
            ),
            method_addon_renderers=(
                {
                    KAPPA_METHOD: lambda: _render_annotator_agreement_section(
                        data,
                        mappings,
                        column_metadata=parse_result["column_metadata"],
                    )
                }
                if rater_unavailability_reason is None
                else None
            ),
            intro_markdown="""
This page computes the main evaluation method families available in the app:
- **Cohen's Kappa** for agreement analysis
- **Classification Metrics** for accuracy and per-class behavior
- **Alt-Test** for viability against human annotators
- **Krippendorff's Alpha** for non-inferiority analysis
- **Classical Krippendorff's Alpha** for direct inter-rater reliability on the selected annotation columns
- **Correlation by Rater** for exploratory redundancy analysis across scales when the template includes `{rater}`
""",
        )
