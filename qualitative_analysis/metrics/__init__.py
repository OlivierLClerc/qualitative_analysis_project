"""
metrics package

This package provides functions for computing various metrics for evaluating
model performance and inter-rater reliability.

Modules:
    - kappa: Functions for computing Cohen's Kappa and related metrics
    - classification: Functions for computing classification metrics
    - alt_test: Functions for performing the Alternative Annotator Test (ALT)
    - visualization: Functions for visualizing metrics
    - utils: Utility functions for computing metrics
"""

# Import all functions from the modules
from qualitative_analysis.metrics.kappa import (
    compute_cohens_kappa,
    compute_all_kappas,
    compute_detailed_kappa_metrics,
    compute_kappa_metrics,
)

from qualitative_analysis.metrics.krippendorff import (
    compute_krippendorff_non_inferiority,
    print_non_inferiority_results,
)

from qualitative_analysis.metrics.classification import (
    ClassMetrics,
    GlobalMetrics,
    ClassificationResults,
    compute_classification_metrics,
    compute_classification_metrics_from_results,
)

from qualitative_analysis.metrics.alt_test import (
    benjamini_yekutieli_correction,
    accuracy_alignment,
    rmse_alignment,
    run_alt_test_general,
    convert_labels,
    run_alt_test_on_results,
)

from qualitative_analysis.metrics.visualization import (
    plot_confusion_matrices,
)

from qualitative_analysis.metrics.utils import (
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
    # Krippendorff metrics
    "compute_krippendorff_non_inferiority",
    "print_non_inferiority_results",
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
