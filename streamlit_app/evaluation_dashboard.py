"""
Shared evaluation dashboard for computing and rendering metrics from a DataFrame.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol

import numpy as np
import pandas as pd
import streamlit as st

from qualitative_analysis.metrics.alt_test import run_alt_test_on_results
from qualitative_analysis.metrics.classification import (
    ClassMetrics,
    compute_classification_metrics,
    compute_classification_metrics_from_results,
)
from qualitative_analysis.metrics.kappa import (
    compute_cohens_kappa,
    compute_detailed_kappa_metrics,
    compute_kappa_metrics,
)
from qualitative_analysis.metrics.utils import compute_majority_vote
from streamlit_app.evaluation_mappings import (
    EvaluationMapping,
    KAPPA_WEIGHT_OPTIONS,
    KRIPP_LEVEL_OPTIONS,
    mapping_label_type_to_alt_test,
    prepare_evaluation_data,
    validate_krippendorff_mapping,
)


class KrippendorffNonInferiorityFn(Protocol):
    def __call__(
        self,
        *,
        detailed_results_df: pd.DataFrame,
        annotation_columns: List[str],
        model_column: str = ...,
        level_of_measurement: str = ...,
        non_inferiority_margin: float = ...,
        n_bootstrap: int = ...,
        confidence_level: float = ...,
        random_seed: int = ...,
        verbose: bool = ...,
    ) -> Dict[str, Dict[str, Any]]: ...


compute_krippendorff_non_inferiority: Optional[KrippendorffNonInferiorityFn]
try:
    from qualitative_analysis.metrics.krippendorff import (
        compute_krippendorff_non_inferiority as _compute_krippendorff_non_inferiority,
    )

    compute_krippendorff_non_inferiority = _compute_krippendorff_non_inferiority
except ModuleNotFoundError:  # pragma: no cover - depends on optional test env deps
    compute_krippendorff_non_inferiority = None

try:
    import krippendorff as krippendorff_lib
except ModuleNotFoundError:  # pragma: no cover - depends on optional test env deps
    krippendorff_lib = None


KAPPA_METHOD = "Cohen's Kappa (Agreement Analysis)"
CLASSIFICATION_METHOD = "Classification Metrics (Balanced Acc, TP%, FP%)"
ALT_METHOD = "Alt-Test (Model Viability)"
KRIPP_METHOD = "Krippendorff's Alpha (Non-Inferiority Test)"
CLASSIC_KRIPP_METHOD = "Krippendorff's Alpha (Classical Agreement)"

METHOD_TO_STATE_KEY = {
    KAPPA_METHOD: "evaluation_results_kappa",
    CLASSIFICATION_METHOD: "evaluation_results_classification",
    ALT_METHOD: "evaluation_results_alt",
    KRIPP_METHOD: "evaluation_results_krippendorff",
    CLASSIC_KRIPP_METHOD: "evaluation_results_classic_krippendorff",
}


def _state_key(method: str, prefix: str) -> str:
    return f"{prefix}{METHOD_TO_STATE_KEY[method]}"


def _widget_key(prefix: str, name: str) -> str:
    return f"{prefix}{name}"


def _mapping_annotation_columns(mapping: EvaluationMapping) -> List[str]:
    columns = mapping.get("annotation_columns")
    if columns:
        return list(columns)
    return list(mapping["human_columns"])


def _format_annotation_columns(mapping: EvaluationMapping) -> str:
    columns = _mapping_annotation_columns(mapping)
    return ", ".join(columns) if columns else "None"


def _display_kappa_weights(weights: Optional[str]) -> str:
    for label, value in KAPPA_WEIGHT_OPTIONS.items():
        if value == weights:
            return label
    return "Unweighted"


def _display_kripp_level(level_of_measurement: Optional[str]) -> str:
    if level_of_measurement in KRIPP_LEVEL_OPTIONS:
        return str(level_of_measurement)
    return "ordinal"


def render_evaluation_dashboard(
    *,
    results_df: pd.DataFrame,
    evaluation_mappings: List[EvaluationMapping],
    ui_key_prefix: str = "",
    comparison_label: str = "LLM field",
    comparison_entity_name: str = "LLM",
    annotation_column_label: str = "Human columns",
    show_comparison_column: bool = True,
    use_mapping_kappa_weights: bool = False,
    use_mapping_kripp_level_of_measurement: bool = False,
    enable_classical_krippendorff: bool = False,
    method_selector_label: str = "Select metric:",
    custom_method_renderers: Optional[Mapping[str, Callable[[], None]]] = None,
    method_addon_renderers: Optional[Mapping[str, Callable[[], None]]] = None,
    classical_kripp_addon_renderer: Optional[Callable[[], None]] = None,
    intro_markdown: Optional[str] = None,
) -> None:
    """
    Render the evaluation dashboard for a prepared DataFrame and mappings.
    """
    if not evaluation_mappings:
        st.info("Add at least one evaluation mapping to compute metrics.")
        return

    if intro_markdown:
        st.markdown(intro_markdown)

    method_options = [
        KAPPA_METHOD,
        CLASSIFICATION_METHOD,
        ALT_METHOD,
        KRIPP_METHOD,
    ]
    if enable_classical_krippendorff:
        method_options.append(CLASSIC_KRIPP_METHOD)
    if custom_method_renderers:
        method_options.extend(custom_method_renderers.keys())

    method_widget_key = _widget_key(ui_key_prefix, "comparison_method")
    if st.session_state.get(method_widget_key) not in method_options:
        st.session_state[method_widget_key] = method_options[0]

    method: str = st.radio(
        method_selector_label,
        tuple(method_options),
        index=0,
        key=method_widget_key,
    )

    if method == KAPPA_METHOD:
        _render_kappa_method(
            results_df,
            evaluation_mappings,
            ui_key_prefix=ui_key_prefix,
            comparison_label=comparison_label,
            comparison_entity_name=comparison_entity_name,
            annotation_column_label=annotation_column_label,
            show_comparison_column=show_comparison_column,
            use_mapping_kappa_weights=use_mapping_kappa_weights,
        )
        if method_addon_renderers and method in method_addon_renderers:
            method_addon_renderers[method]()
    elif method == CLASSIFICATION_METHOD:
        _render_classification_method(
            results_df,
            evaluation_mappings,
            ui_key_prefix=ui_key_prefix,
            comparison_label=comparison_label,
            annotation_column_label=annotation_column_label,
            show_comparison_column=show_comparison_column,
        )
    elif method == ALT_METHOD:
        _render_alt_test_method(
            results_df,
            evaluation_mappings,
            ui_key_prefix=ui_key_prefix,
            comparison_label=comparison_label,
            comparison_entity_name=comparison_entity_name,
            annotation_column_label=annotation_column_label,
            show_comparison_column=show_comparison_column,
        )
    elif method == KRIPP_METHOD:
        _render_krippendorff_method(
            results_df,
            evaluation_mappings,
            ui_key_prefix=ui_key_prefix,
            comparison_label=comparison_label,
            comparison_entity_name=comparison_entity_name,
            annotation_column_label=annotation_column_label,
            show_comparison_column=show_comparison_column,
            use_mapping_kripp_level_of_measurement=use_mapping_kripp_level_of_measurement,
        )
        if method_addon_renderers and method in method_addon_renderers:
            method_addon_renderers[method]()
    elif method == CLASSIC_KRIPP_METHOD:
        _render_classical_krippendorff_method(
            results_df,
            evaluation_mappings,
            ui_key_prefix=ui_key_prefix,
            annotation_column_label=annotation_column_label,
            use_mapping_kripp_level_of_measurement=use_mapping_kripp_level_of_measurement,
        )
        if method_addon_renderers and method in method_addon_renderers:
            method_addon_renderers[method]()
        if classical_kripp_addon_renderer is not None:
            classical_kripp_addon_renderer()
    elif custom_method_renderers and method in custom_method_renderers:
        custom_method_renderers[method]()


def compute_kappa_results(
    results_df: pd.DataFrame,
    evaluation_mappings: List[EvaluationMapping],
    weights: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return [
        _compute_kappa_result(results_df, mapping, weights)
        for mapping in evaluation_mappings
    ]


def compute_classification_results(
    results_df: pd.DataFrame,
    evaluation_mappings: List[EvaluationMapping],
) -> List[Dict[str, Any]]:
    return [
        _compute_classification_result(results_df, mapping)
        for mapping in evaluation_mappings
    ]


def compute_alt_test_results(
    results_df: pd.DataFrame,
    evaluation_mappings: List[EvaluationMapping],
    epsilon_val: float,
    alpha_val: float,
) -> List[Dict[str, Any]]:
    return [
        _compute_alt_test_result(results_df, mapping, epsilon_val, alpha_val)
        for mapping in evaluation_mappings
    ]


def compute_krippendorff_results(
    results_df: pd.DataFrame,
    evaluation_mappings: List[EvaluationMapping],
    level_of_measurement: str,
    non_inferiority_margin: float,
    n_bootstrap: int,
    confidence_level: float,
    use_mapping_levels: bool = False,
) -> List[Dict[str, Any]]:
    return [
        _compute_krippendorff_result(
            results_df,
            mapping,
            (
                str(mapping.get("kripp_level_of_measurement"))
                if use_mapping_levels
                else level_of_measurement
            ),
            non_inferiority_margin,
            n_bootstrap,
            confidence_level,
        )
        for mapping in evaluation_mappings
    ]


def compute_classical_krippendorff_results(
    results_df: pd.DataFrame,
    evaluation_mappings: List[EvaluationMapping],
    level_of_measurement: str,
    use_mapping_levels: bool = False,
) -> List[Dict[str, Any]]:
    return [
        _compute_classical_krippendorff_result(
            results_df,
            mapping,
            (
                str(mapping.get("kripp_level_of_measurement"))
                if use_mapping_levels
                else level_of_measurement
            ),
        )
        for mapping in evaluation_mappings
    ]


def _render_kappa_method(
    results_df: pd.DataFrame,
    evaluation_mappings: List[EvaluationMapping],
    *,
    ui_key_prefix: str,
    comparison_label: str,
    comparison_entity_name: str,
    annotation_column_label: str,
    show_comparison_column: bool,
    use_mapping_kappa_weights: bool,
) -> None:
    st.markdown(
        """
        Analyze agreement between the selected comparison output and human annotators
        for each configured mapping. Use weighting when your labels have an order,
        such as `0, 1, 2`.
        """
    )
    weights: Optional[str]
    if use_mapping_kappa_weights:
        st.caption("Each mapping uses its own correction setting from Step 3.")
        weights = None
    else:
        weights_option = st.radio(
            "Weighting Scheme:",
            ["Unweighted", "Linear", "Quadratic"],
            index=0,
            key=_widget_key(ui_key_prefix, "kappa_weights_option"),
        )
        weights = KAPPA_WEIGHT_OPTIONS[weights_option]

    state_key = _state_key(KAPPA_METHOD, ui_key_prefix)
    if st.button(
        "Compute Agreement Scores",
        key=_widget_key(ui_key_prefix, "compute_agreement_button"),
    ):
        results = []
        for mapping in evaluation_mappings:
            mapping_weights = (
                mapping.get("kappa_weights") if use_mapping_kappa_weights else weights
            )
            results.append(_compute_kappa_result(results_df, mapping, mapping_weights))
        st.session_state[state_key] = results

    cached_results = st.session_state.get(state_key)
    if cached_results:
        _render_kappa_results(
            cached_results,
            None if use_mapping_kappa_weights else weights,
            comparison_label=comparison_label,
            comparison_entity_name=comparison_entity_name,
            annotation_column_label=annotation_column_label,
            show_comparison_column=show_comparison_column,
            ui_key_prefix=ui_key_prefix,
            use_mapping_kappa_weights=use_mapping_kappa_weights,
        )


def _compute_kappa_result(
    results_df: pd.DataFrame,
    mapping: EvaluationMapping,
    weights: Optional[str],
) -> Dict[str, Any]:
    if not mapping["human_columns"]:
        return _skipped_result(mapping, "This mapping has no human annotation columns.")

    prepared = prepare_evaluation_data(results_df, mapping, row_filter="complete")
    if prepared.analysis_data.empty:
        return _skipped_result(
            mapping,
            "No valid rows were available after filtering for this mapping.",
            prepared=prepared,
        )

    summary_df, detailed_kappa_metrics = compute_kappa_metrics(
        detailed_results_df=prepared.analysis_data,
        annotation_columns=mapping["human_columns"],
        labels=prepared.labels,
        kappa_weights=weights,
        coerce_numeric_columns=False,
    )
    if summary_df.empty:
        return _skipped_result(
            mapping,
            "Could not compute kappa metrics from the prepared data.",
            prepared=prepared,
        )

    result_row = summary_df.iloc[0].to_dict()
    return {
        "mapping": mapping,
        "status": "ok",
        "prepared": prepared,
        "summary_df": summary_df,
        "result_row": result_row,
        "detailed_kappa_metrics": detailed_kappa_metrics,
        "has_multiple_runs": result_row.get("n_runs", 1) > 1,
    }


def _render_kappa_results(
    results: List[Dict[str, Any]],
    weights: Optional[str],
    *,
    comparison_label: str,
    comparison_entity_name: str,
    annotation_column_label: str,
    show_comparison_column: bool,
    ui_key_prefix: str,
    use_mapping_kappa_weights: bool,
) -> None:
    st.subheader("Mapping Summary")
    summary_rows = []
    for result in results:
        mapping = result["mapping"]
        annotation_columns = _mapping_annotation_columns(mapping)
        if result["status"] != "ok":
            row = {
                "Mapping": mapping["name"],
                annotation_column_label: len(annotation_columns),
                "Rows used": 0,
                "Headline metric": "Skipped",
                "Status": result["reason"],
            }
            if show_comparison_column:
                row[comparison_label] = mapping["llm_field"]
            if use_mapping_kappa_weights:
                row["Correction"] = _display_kappa_weights(mapping.get("kappa_weights"))
            summary_rows.append(row)
            continue

        result_row = result["result_row"]
        headline_metric = (
            _format_metric(result_row.get("mean_kappa_llm_human"))
            if len(mapping["human_columns"]) > 1
            else _format_metric(result_row.get("kappa_GT_train"))
        )
        row = {
            "Mapping": mapping["name"],
            annotation_column_label: len(annotation_columns),
            "Rows used": len(result["prepared"].analysis_data),
            "Headline metric": headline_metric,
            "Status": "Computed",
        }
        if show_comparison_column:
            row[comparison_label] = mapping["llm_field"]
        if use_mapping_kappa_weights:
            row["Correction"] = _display_kappa_weights(mapping.get("kappa_weights"))
        summary_rows.append(row)

    st.table(pd.DataFrame(summary_rows))

    for result in results:
        mapping = result["mapping"]
        st.markdown(f"#### {mapping['name']}")
        caption_parts = [
            f"{annotation_column_label}: `{_format_annotation_columns(mapping)}`",
            f"Label type: `{mapping['label_type']}`",
        ]
        if show_comparison_column:
            caption_parts.insert(0, f"{comparison_label}: `{mapping['llm_field']}`")
        if use_mapping_kappa_weights:
            caption_parts.append(
                f"Correction: `{_display_kappa_weights(mapping.get('kappa_weights'))}`"
            )
        st.caption(" | ".join(caption_parts))
        _render_preparation_warnings(result)
        if result["status"] != "ok":
            st.warning(result["reason"])
            continue

        result_row = result["result_row"]
        if result["has_multiple_runs"]:
            st.info(
                f"Multiple runs detected: metrics were aggregated across {result_row.get('n_runs', 1)} runs."
            )

        if len(mapping["human_columns"]) > 1:
            st.write(
                f"**Mean {comparison_entity_name}-Human Agreement**: {_format_metric(result_row.get('mean_kappa_llm_human'))}"
            )
            st.write(
                f"**Mean Human-Human Agreement**: {_format_metric(result_row.get('mean_human_human_agreement'))}"
            )
        else:
            human_annotator = mapping["human_columns"][0]
            st.write(
                f"**Cohen's Kappa ({comparison_entity_name} vs {human_annotator})**: {_format_metric(result_row.get('kappa_GT_train'))}"
            )

        scenario_key = next(iter(result["detailed_kappa_metrics"]), None)
        if scenario_key is not None and len(mapping["human_columns"]) > 1:
            kappa_details = result["detailed_kappa_metrics"][scenario_key]
            if not kappa_details["llm_vs_human_df"].empty:
                st.write(f"**{comparison_entity_name} vs Human Annotators**")
                llm_human_display = kappa_details["llm_vs_human_df"].copy()
                llm_human_display["Cohens_Kappa"] = llm_human_display[
                    "Cohens_Kappa"
                ].apply(_format_metric)
                st.table(llm_human_display)

            if not kappa_details["human_vs_human_df"].empty:
                st.write("**Human vs Human Annotators**")
                human_human_display = kappa_details["human_vs_human_df"].copy()
                human_human_display["Cohens_Kappa"] = human_human_display[
                    "Cohens_Kappa"
                ].apply(_format_metric)
                st.table(human_human_display)

        if result["has_multiple_runs"]:
            show_key = _widget_key(ui_key_prefix, f"kappa_per_run_{mapping['id']}")
            if st.checkbox(
                f"Show per-run metrics for {mapping['name']}",
                key=show_key,
            ):
                per_run_weights = (
                    mapping.get("kappa_weights")
                    if use_mapping_kappa_weights
                    else weights
                )
                _render_per_run_kappa(
                    result,
                    per_run_weights,
                    comparison_entity_name=comparison_entity_name,
                )


def _render_per_run_kappa(
    result: Dict[str, Any],
    weights: Optional[str],
    *,
    comparison_entity_name: str,
) -> None:
    prepared = result["prepared"]
    mapping = result["mapping"]
    analysis_data = prepared.analysis_data
    unique_runs = sorted(analysis_data["run"].unique())
    per_run_rows = []

    for run_id in unique_runs:
        run_data = analysis_data[analysis_data["run"] == run_id]
        run_model_predictions = run_data["ModelPrediction"].tolist()
        run_human_annotations = {
            column: run_data[column].tolist() for column in mapping["human_columns"]
        }
        run_ground_truth = compute_majority_vote(run_human_annotations)
        run_kappa = compute_cohens_kappa(
            run_ground_truth,
            run_model_predictions,
            labels=prepared.labels,
            weights=weights,
        )

        row = {
            "Run": run_id,
            "Samples": len(run_data),
            "Kappa vs Ground Truth": _format_metric(run_kappa),
        }
        if len(mapping["human_columns"]) > 1:
            run_detailed_metrics = compute_detailed_kappa_metrics(
                model_predictions=run_model_predictions,
                human_annotations=run_human_annotations,
                labels=prepared.labels,
                kappa_weights=weights,
            )
            row[f"Mean {comparison_entity_name}-Human Kappa"] = _format_metric(
                run_detailed_metrics["mean_llm_human_agreement"]
            )
            row["Mean Human-Human Kappa"] = _format_metric(
                run_detailed_metrics["mean_human_human_agreement"]
            )
        per_run_rows.append(row)

    st.table(pd.DataFrame(per_run_rows))


def _render_classification_method(
    results_df: pd.DataFrame,
    evaluation_mappings: List[EvaluationMapping],
    *,
    ui_key_prefix: str,
    comparison_label: str,
    annotation_column_label: str,
    show_comparison_column: bool,
) -> None:
    st.markdown(
        """
        Analyze detailed classification metrics for each mapping, including accuracy,
        balanced accuracy, and per-class true/false positive behavior.
        """
    )

    state_key = _state_key(CLASSIFICATION_METHOD, ui_key_prefix)
    if st.button(
        "Compute Classification Metrics",
        key=_widget_key(ui_key_prefix, "compute_metrics_button"),
    ):
        st.session_state[state_key] = compute_classification_results(
            results_df, evaluation_mappings
        )

    cached_results = st.session_state.get(state_key)
    if cached_results:
        _render_classification_results(
            cached_results,
            comparison_label=comparison_label,
            annotation_column_label=annotation_column_label,
            show_comparison_column=show_comparison_column,
            ui_key_prefix=ui_key_prefix,
        )


def _compute_classification_result(
    results_df: pd.DataFrame, mapping: EvaluationMapping
) -> Dict[str, Any]:
    if not mapping["human_columns"]:
        return _skipped_result(mapping, "This mapping has no human annotation columns.")

    prepared = prepare_evaluation_data(results_df, mapping, row_filter="complete")
    if prepared.analysis_data.empty:
        return _skipped_result(
            mapping,
            "No valid rows were available after filtering for this mapping.",
            prepared=prepared,
        )

    summary_df = compute_classification_metrics_from_results(
        detailed_results_df=prepared.analysis_data,
        annotation_columns=mapping["human_columns"],
        labels=prepared.labels,
        coerce_numeric_columns=False,
    )
    if summary_df.empty:
        return _skipped_result(
            mapping,
            "Could not compute classification metrics from the prepared data.",
            prepared=prepared,
        )

    train_data = prepared.analysis_data[prepared.analysis_data["split"] == "train"]
    train_model_predictions = train_data["ModelPrediction"].tolist()
    train_human_annotations = {
        column: train_data[column].tolist() for column in mapping["human_columns"]
    }
    direct_train_metrics = compute_classification_metrics(
        model_coding=train_model_predictions,
        human_annotations=train_human_annotations,
        labels=prepared.labels,
    )

    result_row = summary_df.iloc[0].to_dict()
    return {
        "mapping": mapping,
        "status": "ok",
        "prepared": prepared,
        "summary_df": summary_df,
        "result_row": result_row,
        "train_metrics": direct_train_metrics,
        "has_multiple_runs": result_row.get("n_runs", 1) > 1,
    }


def _render_classification_results(
    results: List[Dict[str, Any]],
    *,
    comparison_label: str,
    annotation_column_label: str,
    show_comparison_column: bool,
    ui_key_prefix: str,
) -> None:
    st.subheader("Mapping Summary")
    summary_rows = []
    for result in results:
        mapping = result["mapping"]
        annotation_columns = _mapping_annotation_columns(mapping)
        if result["status"] != "ok":
            row = {
                "Mapping": mapping["name"],
                annotation_column_label: len(annotation_columns),
                "Rows used": 0,
                "Accuracy": "Skipped",
                "Balanced Accuracy": "Skipped",
                "Status": result["reason"],
            }
            if show_comparison_column:
                row[comparison_label] = mapping["llm_field"]
            summary_rows.append(row)
            continue

        result_row = result["result_row"]
        row = {
            "Mapping": mapping["name"],
            annotation_column_label: len(annotation_columns),
            "Rows used": len(result["prepared"].analysis_data),
            "Accuracy": _format_metric(result_row.get("global_accuracy_train")),
            "Balanced Accuracy": _format_metric(result_row.get("global_recall_train")),
            "Status": "Computed",
        }
        if show_comparison_column:
            row[comparison_label] = mapping["llm_field"]
        summary_rows.append(row)

    st.table(pd.DataFrame(summary_rows))

    for result in results:
        mapping = result["mapping"]
        st.markdown(f"#### {mapping['name']}")
        caption_parts = [
            f"{annotation_column_label}: `{_format_annotation_columns(mapping)}`",
            f"Label type: `{mapping['label_type']}`",
        ]
        if show_comparison_column:
            caption_parts.insert(0, f"{comparison_label}: `{mapping['llm_field']}`")
        st.caption(" | ".join(caption_parts))
        _render_preparation_warnings(result)
        if result["status"] != "ok":
            st.warning(result["reason"])
            continue

        result_row = result["result_row"]
        if result["has_multiple_runs"]:
            st.info(
                f"Multiple runs detected: metrics were aggregated across {result_row.get('n_runs', 1)} runs."
            )

        global_metrics_data = [
            {
                "Metric": "Accuracy",
                "Value": _format_metric(result_row.get("global_accuracy_train")),
            },
            {
                "Metric": "Balanced Accuracy (Macro Recall)",
                "Value": _format_metric(result_row.get("global_recall_train")),
            },
            {
                "Metric": "Error Rate",
                "Value": _format_metric(result_row.get("global_error_rate_train")),
            },
            {
                "Metric": "Training samples",
                "Value": int(result_row.get("N_train", 0)),
            },
        ]
        st.table(pd.DataFrame(global_metrics_data))

        st.write("**Per-Class Metrics**")
        per_class_rows = _build_per_class_rows(
            result["train_metrics"]["per_class_metrics"],
            result["prepared"].labels,
        )
        if per_class_rows:
            st.table(pd.DataFrame(per_class_rows))
        else:
            st.info("No per-class metrics are available for this mapping.")

        if result["has_multiple_runs"]:
            show_key = _widget_key(
                ui_key_prefix, f"classification_per_run_{mapping['id']}"
            )
            if st.checkbox(
                f"Show per-run classification metrics for {mapping['name']}",
                key=show_key,
            ):
                _render_per_run_classification(result)


def _render_per_run_classification(result: Dict[str, Any]) -> None:
    prepared = result["prepared"]
    mapping = result["mapping"]
    analysis_data = prepared.analysis_data
    unique_runs = sorted(analysis_data["run"].unique())

    for run_id in unique_runs:
        run_data = analysis_data[analysis_data["run"] == run_id]
        train_run_data = run_data[run_data["split"] == "train"]
        if train_run_data.empty:
            continue

        run_model_predictions = train_run_data["ModelPrediction"].tolist()
        run_human_annotations = {
            column: train_run_data[column].tolist()
            for column in mapping["human_columns"]
        }
        run_metrics = compute_classification_metrics(
            model_coding=run_model_predictions,
            human_annotations=run_human_annotations,
            labels=prepared.labels,
        )

        st.write(f"**Run {run_id}**")
        per_class_rows = _build_per_class_rows(
            run_metrics["per_class_metrics"], prepared.labels
        )
        if per_class_rows:
            st.table(pd.DataFrame(per_class_rows))
        else:
            st.info("No per-class metrics were available for this run.")


def _render_alt_test_method(
    results_df: pd.DataFrame,
    evaluation_mappings: List[EvaluationMapping],
    *,
    ui_key_prefix: str,
    comparison_label: str,
    comparison_entity_name: str,
    annotation_column_label: str,
    show_comparison_column: bool,
) -> None:
    st.markdown(
        """
        Run the Alternative Annotator Test for each mapping. Mappings with fewer than
        3 human columns are skipped because the test requires at least three annotators.
        Float mappings use RMSE alignment automatically; integer and text mappings use exact-match alignment.
        """
    )

    epsilon_val: float = st.number_input(
        "Epsilon (cost-benefit margin)",
        min_value=0.0,
        value=0.1,
        step=0.01,
        key=_widget_key(ui_key_prefix, "alt_test_epsilon"),
    )
    alpha_val: float = st.number_input(
        "Alpha (significance level)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        key=_widget_key(ui_key_prefix, "alt_test_alpha"),
    )

    state_key = _state_key(ALT_METHOD, ui_key_prefix)
    if st.button(
        "Run Alternative Annotator Test",
        key=_widget_key(ui_key_prefix, "run_alt_test_button"),
    ):
        st.session_state[state_key] = compute_alt_test_results(
            results_df, evaluation_mappings, epsilon_val, alpha_val
        )

    cached_results = st.session_state.get(state_key)
    if cached_results:
        _render_alt_test_results(
            cached_results,
            comparison_label=comparison_label,
            comparison_entity_name=comparison_entity_name,
            annotation_column_label=annotation_column_label,
            show_comparison_column=show_comparison_column,
            ui_key_prefix=ui_key_prefix,
        )


def _compute_alt_test_result(
    results_df: pd.DataFrame,
    mapping: EvaluationMapping,
    epsilon_val: float,
    alpha_val: float,
) -> Dict[str, Any]:
    if len(mapping["human_columns"]) < 3:
        return _skipped_result(
            mapping,
            "Alt-Test requires at least 3 human annotation columns.",
        )

    prepared = prepare_evaluation_data(
        results_df,
        mapping,
        row_filter="minimum_humans",
        minimum_human_annotations=3,
    )
    if prepared.analysis_data.empty:
        return _skipped_result(
            mapping,
            "No valid rows were available after filtering for this mapping.",
            prepared=prepared,
        )

    metric, alt_label_type = mapping_label_type_to_alt_test(mapping["label_type"])
    alt_results_df = run_alt_test_on_results(
        detailed_results_df=prepared.analysis_data,
        annotation_columns=mapping["human_columns"],
        labels=prepared.labels,
        epsilon=epsilon_val,
        alpha=alpha_val,
        verbose=False,
        metric=metric,
        label_type=alt_label_type,
    )
    if alt_results_df.empty:
        return _skipped_result(
            mapping,
            "Could not compute ALT test metrics from the prepared data.",
            prepared=prepared,
        )

    aggregated_results = alt_results_df[alt_results_df["run"] == "aggregated"]
    if aggregated_results.empty:
        aggregated_results = alt_results_df.iloc[[0]]
    result_row = aggregated_results.iloc[0].to_dict()

    return {
        "mapping": mapping,
        "status": "ok",
        "prepared": prepared,
        "alt_results_df": alt_results_df,
        "result_row": result_row,
        "alpha_value": alpha_val,
        "has_multiple_runs": result_row.get("n_runs", 1) > 1,
    }


def _render_alt_test_results(
    results: List[Dict[str, Any]],
    *,
    comparison_label: str,
    comparison_entity_name: str,
    annotation_column_label: str,
    show_comparison_column: bool,
    ui_key_prefix: str,
) -> None:
    st.subheader("Mapping Summary")
    summary_rows = []
    for result in results:
        mapping = result["mapping"]
        if result["status"] != "ok":
            summary_rows.append(
                {
                    "Mapping": mapping["name"],
                    "Rows used": 0,
                    "Winning Rate": "Skipped",
                    "Average Advantage": "Skipped",
                    "Status": result["reason"],
                }
            )
            continue

        result_row = result["result_row"]
        summary_rows.append(
            {
                "Mapping": mapping["name"],
                "Rows used": len(result["prepared"].analysis_data),
                "Winning Rate": _format_metric(result_row.get("winning_rate_train")),
                "Average Advantage": _format_metric(
                    result_row.get("avg_adv_prob_train")
                ),
                "Status": (
                    "Passed"
                    if result_row.get("passed_alt_test_train", False)
                    else "Did not pass"
                ),
            }
        )

    st.table(pd.DataFrame(summary_rows))

    for result in results:
        mapping = result["mapping"]
        st.markdown(f"#### {mapping['name']}")
        caption_parts = [
            f"{annotation_column_label}: `{_format_annotation_columns(mapping)}`",
            f"Label type: `{mapping['label_type']}`",
        ]
        if show_comparison_column:
            caption_parts.insert(0, f"{comparison_label}: `{mapping['llm_field']}`")
        st.caption(" | ".join(caption_parts))
        _render_preparation_warnings(result)
        if result["status"] != "ok":
            st.warning(result["reason"])
            continue

        result_row = result["result_row"]
        if result["has_multiple_runs"]:
            st.info(
                f"Multiple runs detected: metrics were aggregated across {result_row.get('n_runs', 1)} runs."
            )

        p_values_train = result_row.get("p_values_train", [])
        winning_rate = result_row.get("winning_rate_train", 0)
        passed_test = result_row.get("passed_alt_test_train", False)
        avg_adv_prob = result_row.get("avg_adv_prob_train", 0)

        if p_values_train:
            st.table(
                pd.DataFrame(
                    _build_alt_test_rows(
                        mapping["human_columns"],
                        p_values_train,
                        avg_adv_prob,
                        result["alpha_value"],
                        comparison_entity_name,
                    )
                )
            )

        st.write(f"**Winning Rate (omega)**: {_format_metric(winning_rate)}")
        st.write(
            f"**Average {comparison_entity_name} Advantage (rho)**: {_format_metric(avg_adv_prob)}"
        )

        if passed_test:
            st.success(
                "The selected comparison output passed the Alt-Test for this mapping."
            )
        else:
            st.warning(
                "The selected comparison output did not pass the Alt-Test for this mapping."
            )

        if result["has_multiple_runs"]:
            show_key = _widget_key(ui_key_prefix, f"alt_per_run_{mapping['id']}")
            if st.checkbox(
                f"Show per-run Alt-Test metrics for {mapping['name']}",
                key=show_key,
            ):
                _render_per_run_alt_test(
                    result,
                    comparison_entity_name=comparison_entity_name,
                )


def _render_per_run_alt_test(
    result: Dict[str, Any], *, comparison_entity_name: str
) -> None:
    alt_results_df = result["alt_results_df"]
    mapping = result["mapping"]
    individual_runs = alt_results_df[alt_results_df["run"] != "aggregated"]

    if individual_runs.empty:
        st.info("No per-run Alt-Test data is available.")
        return

    for _, run_row in individual_runs.iterrows():
        run_id = run_row["run"]
        st.write(f"**Run {run_id}**")
        p_values_train = run_row.get("p_values_train", [])
        avg_adv_prob = run_row.get("avg_adv_prob_train", 0)
        if p_values_train:
            st.table(
                pd.DataFrame(
                    _build_alt_test_rows(
                        mapping["human_columns"],
                        p_values_train,
                        avg_adv_prob,
                        result["alpha_value"],
                        comparison_entity_name,
                    )
                )
            )
        st.write(
            f"Winning Rate (omega): {_format_metric(run_row.get('winning_rate_train'))}"
        )
        st.write(
            f"Average {comparison_entity_name} Advantage (rho): {_format_metric(run_row.get('avg_adv_prob_train'))}"
        )


def _compute_classical_krippendorff_alpha(
    frame: pd.DataFrame,
    annotation_columns: List[str],
    level_of_measurement: str,
) -> Any:
    if krippendorff_lib is None:
        return np.nan

    data = frame[annotation_columns].to_numpy(dtype=float).T
    annotator_mask = ~np.all(np.isnan(data), axis=1)
    data = data[annotator_mask, :]
    if data.shape[0] < 2:
        return np.nan

    valid_items_mask = ~np.all(np.isnan(data), axis=0)
    data = data[:, valid_items_mask]
    if data.shape[1] < 2:
        return np.nan

    try:
        return float(
            krippendorff_lib.alpha(data, level_of_measurement=level_of_measurement)
        )
    except Exception:
        return np.nan


def _compute_classical_krippendorff_result(
    results_df: pd.DataFrame,
    mapping: EvaluationMapping,
    level_of_measurement: str,
) -> Dict[str, Any]:
    if krippendorff_lib is None:
        return _skipped_result(
            mapping,
            "Classical Krippendorff support is unavailable because the optional dependency is not installed.",
        )

    if len(mapping["human_columns"]) < 2:
        return _skipped_result(
            mapping,
            "Classical Krippendorff's alpha requires at least 2 annotation columns.",
        )

    if mapping["label_type"] == "Text" and level_of_measurement != "nominal":
        return _skipped_result(
            mapping,
            "Text mappings can only be used with nominal Krippendorff measurement.",
        )

    prepared = prepare_evaluation_data(
        results_df,
        mapping,
        row_filter="minimum_humans",
        minimum_human_annotations=2,
        encode_text_for_krippendorff=mapping["label_type"] == "Text",
    )
    if prepared.analysis_data.empty:
        return _skipped_result(
            mapping,
            "No valid rows were available after filtering for this mapping.",
            prepared=prepared,
        )

    alpha_overall = _compute_classical_krippendorff_alpha(
        prepared.analysis_data,
        mapping["human_columns"],
        level_of_measurement,
    )
    if pd.isna(alpha_overall):
        return _skipped_result(
            mapping,
            "Could not compute classical Krippendorff's alpha from the prepared data.",
            prepared=prepared,
        )

    per_run_rows = []
    if "run" in prepared.analysis_data.columns:
        for run_id in sorted(prepared.analysis_data["run"].dropna().unique()):
            run_data = prepared.analysis_data[prepared.analysis_data["run"] == run_id]
            alpha_value = _compute_classical_krippendorff_alpha(
                run_data,
                mapping["human_columns"],
                level_of_measurement,
            )
            per_run_rows.append(
                {
                    "Run": run_id,
                    "Rows": len(run_data),
                    "Alpha": alpha_value,
                }
            )

    valid_per_run_alphas = [
        row["Alpha"] for row in per_run_rows if not pd.isna(row["Alpha"])
    ]
    has_multiple_runs = len(per_run_rows) > 1

    return {
        "mapping": mapping,
        "status": "ok",
        "prepared": prepared,
        "alpha_overall": alpha_overall,
        "per_run_rows": per_run_rows,
        "has_multiple_runs": has_multiple_runs,
        "alpha_mean": (
            float(np.mean(valid_per_run_alphas))
            if valid_per_run_alphas
            else float(alpha_overall)
        ),
        "alpha_std": (
            float(np.std(valid_per_run_alphas))
            if len(valid_per_run_alphas) > 1
            else 0.0
        ),
        "n_runs": len(valid_per_run_alphas) if valid_per_run_alphas else 1,
        "level_of_measurement": level_of_measurement,
    }


def _render_classical_krippendorff_method(
    results_df: pd.DataFrame,
    evaluation_mappings: List[EvaluationMapping],
    *,
    ui_key_prefix: str,
    annotation_column_label: str,
    use_mapping_kripp_level_of_measurement: bool,
) -> None:
    st.markdown(
        """
        Compute the classical inter-rater Krippendorff's alpha for each mapping using
        the selected annotation columns only. This does not run the non-inferiority test.
        """
    )

    if use_mapping_kripp_level_of_measurement:
        st.caption("Each mapping uses its own Krippendorff level from Step 3.")
        level_of_measurement = "ordinal"
    else:
        level_of_measurement = st.radio(
            "Level of Measurement:",
            KRIPP_LEVEL_OPTIONS,
            index=0,
            key=_widget_key(ui_key_prefix, "classic_kripp_level_measurement"),
        )

    state_key = _state_key(CLASSIC_KRIPP_METHOD, ui_key_prefix)
    if st.button(
        "Compute Classical Krippendorff's Alpha",
        key=_widget_key(ui_key_prefix, "run_classic_kripp_button"),
    ):
        st.session_state[state_key] = compute_classical_krippendorff_results(
            results_df,
            evaluation_mappings,
            level_of_measurement,
            use_mapping_levels=use_mapping_kripp_level_of_measurement,
        )

    cached_results = st.session_state.get(state_key)
    if cached_results:
        _render_classical_krippendorff_results(
            cached_results,
            annotation_column_label=annotation_column_label,
            ui_key_prefix=ui_key_prefix,
            use_mapping_kripp_level_of_measurement=use_mapping_kripp_level_of_measurement,
        )


def _render_classical_krippendorff_results(
    results: List[Dict[str, Any]],
    *,
    annotation_column_label: str,
    ui_key_prefix: str,
    use_mapping_kripp_level_of_measurement: bool,
) -> None:
    st.subheader("Mapping Summary")
    summary_rows = []
    for result in results:
        mapping = result["mapping"]
        annotation_columns = _mapping_annotation_columns(mapping)
        row = {
            "Mapping": mapping["name"],
            annotation_column_label: len(annotation_columns),
            "Rows used": (
                len(result["prepared"].analysis_data)
                if result.get("prepared") is not None
                else 0
            ),
            "Alpha": (
                _format_metric(result.get("alpha_mean"))
                if result["status"] == "ok"
                else "Skipped"
            ),
            "Status": ("Computed" if result["status"] == "ok" else result["reason"]),
        }
        if use_mapping_kripp_level_of_measurement:
            row["Measurement"] = _display_kripp_level(
                mapping.get("kripp_level_of_measurement")
            )
        summary_rows.append(row)

    st.table(pd.DataFrame(summary_rows))

    for result in results:
        mapping = result["mapping"]
        st.markdown(f"#### {mapping['name']}")
        caption_parts = [
            f"{annotation_column_label}: `{_format_annotation_columns(mapping)}`",
            f"Label type: `{mapping['label_type']}`",
        ]
        if use_mapping_kripp_level_of_measurement:
            caption_parts.append(
                "Measurement: "
                f"`{_display_kripp_level(mapping.get('kripp_level_of_measurement'))}`"
            )
        st.caption(" | ".join(caption_parts))
        _render_preparation_warnings(result)

        if result["status"] != "ok":
            st.warning(result["reason"])
            continue

        if result["has_multiple_runs"]:
            st.write(
                f"**Mean alpha across runs**: {_format_metric(result['alpha_mean'])} +/- {_format_metric(result['alpha_std'])}"
            )
            st.write(
                f"**Pooled alpha across all rows**: {_format_metric(result['alpha_overall'])}"
            )
        else:
            st.write(
                f"**Krippendorff's alpha**: {_format_metric(result['alpha_overall'])}"
            )

        if result["has_multiple_runs"]:
            show_key = _widget_key(
                ui_key_prefix, f"classic_kripp_per_run_{mapping['id']}"
            )
            if st.checkbox(
                f"Show per-run classical Krippendorff metrics for {mapping['name']}",
                key=show_key,
            ):
                per_run_display = pd.DataFrame(result["per_run_rows"]).copy()
                if not per_run_display.empty:
                    per_run_display["Alpha"] = per_run_display["Alpha"].apply(
                        _format_metric
                    )
                st.table(per_run_display)


def _render_krippendorff_method(
    results_df: pd.DataFrame,
    evaluation_mappings: List[EvaluationMapping],
    *,
    ui_key_prefix: str,
    comparison_label: str,
    comparison_entity_name: str,
    annotation_column_label: str,
    show_comparison_column: bool,
    use_mapping_kripp_level_of_measurement: bool,
) -> None:
    st.markdown(
        """
        Run Krippendorff's alpha non-inferiority analysis for each mapping.
        Text mappings are only valid with **nominal** measurement. Mappings with fewer than
        3 human columns are skipped automatically.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        if use_mapping_kripp_level_of_measurement:
            st.caption("Each mapping uses its own Krippendorff level from Step 3.")
            level_of_measurement = "ordinal"
        else:
            level_of_measurement = st.radio(
                "Level of Measurement:",
                KRIPP_LEVEL_OPTIONS,
                index=0,
                key=_widget_key(ui_key_prefix, "kripp_level_measurement"),
            )
        non_inferiority_margin: float = st.number_input(
            "Non-inferiority margin (delta)",
            min_value=-1.0,
            max_value=0.0,
            value=-0.05,
            step=0.01,
            key=_widget_key(ui_key_prefix, "kripp_margin"),
        )

    with col2:
        n_bootstrap: int = st.number_input(
            "Bootstrap samples",
            min_value=100,
            max_value=10000,
            value=2000,
            step=100,
            key=_widget_key(ui_key_prefix, "kripp_bootstrap"),
        )
        confidence_level: float = st.number_input(
            "Confidence level (%)",
            min_value=80.0,
            max_value=99.0,
            value=90.0,
            step=1.0,
            key=_widget_key(ui_key_prefix, "kripp_confidence"),
        )

    state_key = _state_key(KRIPP_METHOD, ui_key_prefix)
    if st.button(
        "Run Krippendorff's Alpha Test",
        key=_widget_key(ui_key_prefix, "run_kripp_test_button"),
    ):
        st.session_state[state_key] = compute_krippendorff_results(
            results_df,
            evaluation_mappings,
            level_of_measurement,
            non_inferiority_margin,
            n_bootstrap,
            confidence_level,
            use_mapping_levels=use_mapping_kripp_level_of_measurement,
        )

    cached_results = st.session_state.get(state_key)
    if cached_results:
        _render_krippendorff_results(
            cached_results,
            confidence_level,
            comparison_label=comparison_label,
            comparison_entity_name=comparison_entity_name,
            annotation_column_label=annotation_column_label,
            show_comparison_column=show_comparison_column,
            ui_key_prefix=ui_key_prefix,
            use_mapping_kripp_level_of_measurement=use_mapping_kripp_level_of_measurement,
        )


def _compute_krippendorff_result(
    results_df: pd.DataFrame,
    mapping: EvaluationMapping,
    level_of_measurement: str,
    non_inferiority_margin: float,
    n_bootstrap: int,
    confidence_level: float,
) -> Dict[str, Any]:
    if compute_krippendorff_non_inferiority is None:
        return _skipped_result(
            mapping,
            "Krippendorff support is unavailable because the optional dependency is not installed.",
        )

    validation_error = validate_krippendorff_mapping(mapping, level_of_measurement)
    if validation_error:
        return _skipped_result(mapping, validation_error)

    prepared = prepare_evaluation_data(
        results_df,
        mapping,
        row_filter="minimum_humans",
        minimum_human_annotations=3,
        encode_text_for_krippendorff=mapping["label_type"] == "Text",
    )
    if prepared.analysis_data.empty:
        return _skipped_result(
            mapping,
            "No valid rows were available after filtering for this mapping.",
            prepared=prepared,
        )

    kripp_results = compute_krippendorff_non_inferiority(
        detailed_results_df=prepared.analysis_data,
        annotation_columns=mapping["human_columns"],
        model_column="ModelPrediction",
        level_of_measurement=level_of_measurement,
        non_inferiority_margin=non_inferiority_margin,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        verbose=False,
    )
    if not kripp_results:
        return _skipped_result(
            mapping,
            "Could not compute Krippendorff's alpha from the prepared data.",
            prepared=prepared,
        )

    scenario_key = next(iter(kripp_results))
    scenario_results = kripp_results[scenario_key]
    agg_metrics = scenario_results["aggregated_metrics"]

    return {
        "mapping": mapping,
        "status": "ok",
        "prepared": prepared,
        "kripp_results": kripp_results,
        "aggregated_metrics": agg_metrics,
    }


def _render_krippendorff_results(
    results: List[Dict[str, Any]],
    confidence_level: float,
    *,
    comparison_label: str,
    comparison_entity_name: str,
    annotation_column_label: str,
    show_comparison_column: bool,
    ui_key_prefix: str,
    use_mapping_kripp_level_of_measurement: bool,
) -> None:
    st.subheader("Mapping Summary")
    summary_rows = []
    for result in results:
        mapping = result["mapping"]
        if result["status"] != "ok":
            row = {
                "Mapping": mapping["name"],
                "Rows used": 0,
                "Human alpha": "Skipped",
                f"{comparison_entity_name} alpha": "Skipped",
                "Status": result["reason"],
            }
            if use_mapping_kripp_level_of_measurement:
                row["Measurement"] = _display_kripp_level(
                    mapping.get("kripp_level_of_measurement")
                )
            summary_rows.append(row)
            continue

        agg_metrics = result["aggregated_metrics"]
        row = {
            "Mapping": mapping["name"],
            "Rows used": len(result["prepared"].analysis_data),
            "Human alpha": _format_metric(agg_metrics.get("alpha_human_trios_mean")),
            f"{comparison_entity_name} alpha": _format_metric(
                agg_metrics.get("alpha_model_trios_mean")
            ),
            "Status": (
                "Non-inferior"
                if agg_metrics.get("n_non_inferior", 0) == agg_metrics.get("n_runs", 0)
                else "See details"
            ),
        }
        if use_mapping_kripp_level_of_measurement:
            row["Measurement"] = _display_kripp_level(
                mapping.get("kripp_level_of_measurement")
            )
        summary_rows.append(row)

    st.table(pd.DataFrame(summary_rows))

    for result in results:
        mapping = result["mapping"]
        st.markdown(f"#### {mapping['name']}")
        caption_parts = [
            f"{annotation_column_label}: `{_format_annotation_columns(mapping)}`",
            f"Label type: `{mapping['label_type']}`",
        ]
        if show_comparison_column:
            caption_parts.insert(0, f"{comparison_label}: `{mapping['llm_field']}`")
        if use_mapping_kripp_level_of_measurement:
            caption_parts.append(
                "Measurement: "
                f"`{_display_kripp_level(mapping.get('kripp_level_of_measurement'))}`"
            )
        st.caption(" | ".join(caption_parts))
        _render_preparation_warnings(result)
        if result["status"] != "ok":
            st.warning(result["reason"])
            continue

        agg_metrics = result["aggregated_metrics"]
        n_runs = agg_metrics.get("n_runs", 1)
        if n_runs > 1:
            st.info(
                f"Multiple runs detected: Krippendorff metrics were aggregated across {n_runs} runs."
            )

        st.write(
            f"**Human trios alpha**: {_format_metric(agg_metrics.get('alpha_human_trios_mean'))} +/- {_format_metric(agg_metrics.get('alpha_human_trios_std'))}"
        )
        st.write(
            f"**{comparison_entity_name} trios alpha**: {_format_metric(agg_metrics.get('alpha_model_trios_mean'))} +/- {_format_metric(agg_metrics.get('alpha_model_trios_std'))}"
        )
        st.write(
            f"**Delta = {comparison_entity_name} - human**: {_format_signed_metric(agg_metrics.get('difference_mean'))} +/- {_format_metric(agg_metrics.get('difference_std'))}"
        )
        st.write(
            f"**{confidence_level:.0f}% CI**: [{_format_metric(agg_metrics.get('ci_lower_mean'))}, {_format_metric(agg_metrics.get('ci_upper_mean'))}]"
        )
        st.write(
            f"**Non-inferiority demonstrated in {agg_metrics.get('n_non_inferior', 0)}/{n_runs} runs**"
        )

        if agg_metrics.get("n_non_inferior", 0) == n_runs:
            st.success("Non-inferiority was demonstrated across all runs.")
        elif agg_metrics.get("n_non_inferior", 0) > 0:
            st.warning("Non-inferiority was demonstrated in some runs but not all.")
        else:
            st.error("Non-inferiority was not demonstrated for this mapping.")

        if n_runs > 1:
            show_key = _widget_key(ui_key_prefix, f"kripp_per_run_{mapping['id']}")
            if st.checkbox(
                f"Show per-run Krippendorff metrics for {mapping['name']}",
                key=show_key,
            ):
                _render_per_run_krippendorff(
                    result,
                    confidence_level,
                    comparison_entity_name=comparison_entity_name,
                )


def _render_per_run_krippendorff(
    result: Dict[str, Any],
    confidence_level: float,
    *,
    comparison_entity_name: str,
) -> None:
    kripp_results = result["kripp_results"]
    scenario_key = next(iter(kripp_results))
    scenario_results = kripp_results[scenario_key]

    per_run_rows = []
    for run_result in scenario_results["run_results"]:
        per_run_rows.append(
            {
                "Run": run_result["run"],
                "Human alpha": _format_metric(run_result["alpha_human_groups"]),
                f"{comparison_entity_name} alpha": _format_metric(
                    run_result["alpha_model_groups"]
                ),
                "Delta": _format_signed_metric(run_result["difference"]),
                f"{confidence_level:.0f}% CI": (
                    f"[{_format_metric(run_result['ci_lower'])}, {_format_metric(run_result['ci_upper'])}]"
                ),
                "Non-inferior": run_result["non_inferiority_demonstrated"],
            }
        )

    st.table(pd.DataFrame(per_run_rows))


def _build_per_class_rows(
    per_class_metrics: Mapping[Any, Mapping[str, ClassMetrics]],
    labels: List[Any],
) -> List[Dict[str, Any]]:
    rows = []
    for label in labels:
        if label not in per_class_metrics:
            continue

        class_metrics = per_class_metrics[label]["model"]
        tp_count = int(class_metrics["correct_count"])
        fp_count = int(class_metrics["false_positives"])
        total_predictions = tp_count + fp_count
        fp_percentage = fp_count / total_predictions if total_predictions > 0 else 0.0
        rows.append(
            {
                "Class": label,
                "TP% (Recall)": f"{class_metrics['recall']:.2%}",
                "FP%": f"{fp_percentage:.2%}",
                "True Positives": tp_count,
                "False Positives": fp_count,
            }
        )
    return rows


def _build_alt_test_rows(
    annotators: List[str],
    p_values: List[Any],
    avg_adv_prob: float,
    alpha_value: float,
    comparison_entity_name: str,
) -> List[Dict[str, Any]]:
    rows = []
    for index, annotator in enumerate(annotators):
        if index >= len(p_values):
            continue
        p_value = p_values[index]
        reject_h0 = p_value < alpha_value if not pd.isna(p_value) else False
        rows.append(
            {
                "Annotator": annotator,
                "p-value": _format_metric(p_value),
                "RejectH0?": reject_h0,
                f"rho_f ({comparison_entity_name} advantage)": _format_metric(
                    avg_adv_prob
                ),
                "rho_h (Human advantage)": _format_metric(1 - avg_adv_prob),
            }
        )
    return rows


def _render_preparation_warnings(result: Dict[str, Any]) -> None:
    prepared = result.get("prepared")
    if prepared is None:
        return

    shown = set()
    for warning in prepared.warnings:
        if warning not in shown:
            st.warning(warning)
            shown.add(warning)


def _skipped_result(
    mapping: EvaluationMapping,
    reason: str,
    prepared: Optional[Any] = None,
) -> Dict[str, Any]:
    return {
        "mapping": mapping,
        "status": "skipped",
        "reason": reason,
        "prepared": prepared,
    }


def _format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def _format_signed_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):+.4f}"
