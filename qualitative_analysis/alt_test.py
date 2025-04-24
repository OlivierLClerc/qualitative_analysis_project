"""
alt_test.py

This module provides utility functions for performing the Alternative Annotator Test (alt-test)
on data with model predictions and human annotations.

This is a backward compatibility module that imports from the metrics package.
New code should import directly from the metrics package.

For example, instead of:
    from qualitative_analysis.alt_test import run_alt_test_general

Use:
    from qualitative_analysis.metrics import run_alt_test_general
    or
    from qualitative_analysis.metrics.alt_test import run_alt_test_general
"""

# Import all functions from the metrics.alt_test module
from qualitative_analysis.metrics.alt_test import (
    convert_labels,
    benjamini_yekutieli_correction,
    accuracy_alignment,
    rmse_alignment,
    run_alt_test_general,
    run_alt_test_on_results,
)

# For backward compatibility
__all__ = [
    "convert_labels",
    "benjamini_yekutieli_correction",
    "accuracy_alignment",
    "rmse_alignment",
    "run_alt_test_general",
    "run_alt_test_on_results",
]

# Deprecation warning
import warnings

warnings.warn(
    "The 'qualitative_analysis.alt_test' module is deprecated. "
    "Please import directly from 'qualitative_analysis.metrics.alt_test' instead.",
    DeprecationWarning,
    stacklevel=2,
)
