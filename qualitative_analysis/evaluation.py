"""
evaluation.py

This module provides functions for computing various metrics for evaluating
model performance and inter-rater reliability.

This is a backward compatibility module that imports from the metrics package.
New code should import directly from the metrics package.

For example, instead of:
    from qualitative_analysis.evaluation import compute_cohens_kappa

Use:
    from qualitative_analysis.metrics import compute_cohens_kappa
    or
    from qualitative_analysis.metrics.kappa import compute_cohens_kappa
"""

# Import all functions from the metrics package
from qualitative_analysis.metrics import (
    # Kappa metrics
    compute_cohens_kappa,
    compute_all_kappas,
    compute_detailed_kappa_metrics,
    compute_kappa_metrics,
    # Classification metrics
    ClassMetrics,
    GlobalMetrics,
    ClassificationResults,
    compute_classification_metrics,
    compute_classification_metrics_from_results,
    # ALT test
    benjamini_yekutieli_correction,
    accuracy_alignment,
    rmse_alignment,
    run_alt_test_general,
    convert_labels,
    run_alt_test_on_results,
    # Visualization
    plot_confusion_matrices,
    # Utils
    compute_human_accuracies,
    compute_majority_vote,
)

# For backward compatibility
__all__ = [
    # Kappa metrics
    "compute_cohens_kappa",
    "compute_all_kappas",
    "compute_detailed_kappa_metrics",
    "compute_kappa_metrics",
    # Classification metrics
    "ClassMetrics",
    "GlobalMetrics",
    "ClassificationResults",
    "compute_classification_metrics",
    "compute_classification_metrics_from_results",
    # ALT test
    "benjamini_yekutieli_correction",
    "accuracy_alignment",
    "rmse_alignment",
    "run_alt_test_general",
    "convert_labels",
    "run_alt_test_on_results",
    # Visualization
    "plot_confusion_matrices",
    # Utils
    "compute_human_accuracies",
    "compute_majority_vote",
]

# Deprecation warning
import warnings

warnings.warn(
    "The 'qualitative_analysis.evaluation' module is deprecated. "
    "Please import directly from 'qualitative_analysis.metrics' instead.",
    DeprecationWarning,
    stacklevel=2,
)
