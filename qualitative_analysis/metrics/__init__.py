"""
Metrics package exports, loaded lazily to avoid importing optional dependencies
until they are actually needed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Kappa metrics
    "compute_cohens_kappa": (
        "qualitative_analysis.metrics.kappa",
        "compute_cohens_kappa",
    ),
    "compute_all_kappas": ("qualitative_analysis.metrics.kappa", "compute_all_kappas"),
    "compute_detailed_kappa_metrics": (
        "qualitative_analysis.metrics.kappa",
        "compute_detailed_kappa_metrics",
    ),
    "compute_kappa_metrics": (
        "qualitative_analysis.metrics.kappa",
        "compute_kappa_metrics",
    ),
    # Krippendorff metrics
    "compute_krippendorff_non_inferiority": (
        "qualitative_analysis.metrics.krippendorff",
        "compute_krippendorff_non_inferiority",
    ),
    "print_non_inferiority_results": (
        "qualitative_analysis.metrics.krippendorff",
        "print_non_inferiority_results",
    ),
    # Classification metrics
    "ClassMetrics": ("qualitative_analysis.metrics.classification", "ClassMetrics"),
    "GlobalMetrics": ("qualitative_analysis.metrics.classification", "GlobalMetrics"),
    "ClassificationResults": (
        "qualitative_analysis.metrics.classification",
        "ClassificationResults",
    ),
    "compute_classification_metrics": (
        "qualitative_analysis.metrics.classification",
        "compute_classification_metrics",
    ),
    "compute_classification_metrics_from_results": (
        "qualitative_analysis.metrics.classification",
        "compute_classification_metrics_from_results",
    ),
    # ALT test
    "benjamini_yekutieli_correction": (
        "qualitative_analysis.metrics.alt_test",
        "benjamini_yekutieli_correction",
    ),
    "accuracy_alignment": (
        "qualitative_analysis.metrics.alt_test",
        "accuracy_alignment",
    ),
    "rmse_alignment": ("qualitative_analysis.metrics.alt_test", "rmse_alignment"),
    "run_alt_test_general": (
        "qualitative_analysis.metrics.alt_test",
        "run_alt_test_general",
    ),
    "convert_labels": ("qualitative_analysis.metrics.alt_test", "convert_labels"),
    "run_alt_test_on_results": (
        "qualitative_analysis.metrics.alt_test",
        "run_alt_test_on_results",
    ),
    # Visualization
    "plot_confusion_matrices": (
        "qualitative_analysis.metrics.visualization",
        "plot_confusion_matrices",
    ),
    # Utils
    "compute_human_accuracies": (
        "qualitative_analysis.metrics.utils",
        "compute_human_accuracies",
    ),
    "compute_majority_vote": (
        "qualitative_analysis.metrics.utils",
        "compute_majority_vote",
    ),
}


__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(
            f"module 'qualitative_analysis.metrics' has no attribute {name!r}"
        )

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
