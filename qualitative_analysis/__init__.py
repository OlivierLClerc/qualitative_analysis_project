"""
Top-level package exports for qualitative analysis, loaded lazily.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Data processing
    "load_data": ("qualitative_analysis.data_processing", "load_data"),
    "clean_and_normalize": (
        "qualitative_analysis.data_processing",
        "clean_and_normalize",
    ),
    "sanitize_dataframe": (
        "qualitative_analysis.data_processing",
        "sanitize_dataframe",
    ),
    "select_and_rename_columns": (
        "qualitative_analysis.data_processing",
        "select_and_rename_columns",
    ),
    "load_results_from_csv": (
        "qualitative_analysis.data_processing",
        "load_results_from_csv",
    ),
    # Metrics
    "compute_cohens_kappa": ("qualitative_analysis.metrics", "compute_cohens_kappa"),
    "compute_all_kappas": ("qualitative_analysis.metrics", "compute_all_kappas"),
    "compute_classification_metrics": (
        "qualitative_analysis.metrics",
        "compute_classification_metrics",
    ),
    "compute_classification_metrics_from_results": (
        "qualitative_analysis.metrics",
        "compute_classification_metrics_from_results",
    ),
    "compute_detailed_kappa_metrics": (
        "qualitative_analysis.metrics",
        "compute_detailed_kappa_metrics",
    ),
    "plot_confusion_matrices": (
        "qualitative_analysis.metrics",
        "plot_confusion_matrices",
    ),
    "run_alt_test_general": ("qualitative_analysis.metrics", "run_alt_test_general"),
    # Model interaction
    "LLMClient": ("qualitative_analysis.model_interaction", "LLMClient"),
    "OpenAILLMClient": (
        "qualitative_analysis.model_interaction",
        "OpenAILLMClient",
    ),
    "TogetherLLMClient": (
        "qualitative_analysis.model_interaction",
        "TogetherLLMClient",
    ),
    "get_llm_client": ("qualitative_analysis.model_interaction", "get_llm_client"),
    # Notebooks functions
    "process_general_verbatims": (
        "qualitative_analysis.notebooks_functions",
        "process_general_verbatims",
    ),
    # Parsing
    "parse_llm_response": ("qualitative_analysis.parsing", "parse_llm_response"),
    "extract_code_from_response": (
        "qualitative_analysis.parsing",
        "extract_code_from_response",
    ),
    "extract_global_validity": (
        "qualitative_analysis.parsing",
        "extract_global_validity",
    ),
    # Cost estimation
    "openai_api_calculate_cost": (
        "qualitative_analysis.cost_estimation",
        "openai_api_calculate_cost",
    ),
    # Logging
    "calculate_and_log": ("qualitative_analysis.logging", "calculate_and_log"),
    # Scenario runner
    "run_scenarios": ("qualitative_analysis.scenario_runner", "run_scenarios"),
}


__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'qualitative_analysis' has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
