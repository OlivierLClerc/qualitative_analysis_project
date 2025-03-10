"""
alt_test_module.py

This module provides utility functions for performing the Alternative Annotator Test (alt-test)
on data with model predictions and human annotations. It supports two types of alignment metrics:
    - "accuracy": for discrete classification tasks (default).
    - "rmse": for continuous labels (using negative RMSE so that higher is better).

It also includes functionality for multiple-comparison correction using the Benjamini–Yekutieli procedure,
and computes the winning rate and average advantage probability.

Dependencies:
    - pandas
    - numpy
    - scipy.stats (ttest_1samp)
    - math

Functions:
    - benjamini_yekutieli_correction(pvals, alpha=0.05)
    - accuracy_alignment(source, others)
    - rmse_alignment(source, others)
    - run_alt_test_general(...)
"""

import pandas as pd
from typing import List, Dict, Any, Callable
import numpy as np
from scipy.stats import ttest_1samp


def convert_labels(labels: List[Any], label_type: str = "auto") -> List[Any]:
    """
    Convert a list of labels to the specified type.

    Parameters
    ----------
    labels : List[Any]
        The list of labels to convert.
    label_type : str, optional
        The type to convert the labels to. Options are:
        - "int": Convert all labels to integers
        - "str": Convert all labels to strings
        - "auto": Infer the best type (default)

    Returns
    -------
    List[Any]
        The list of converted labels.
    """
    if label_type == "int":
        # Convert all labels to integers
        return [
            int(label) if label != -1 and not pd.isna(label) else label
            for label in labels
        ]
    elif label_type == "str":
        # Convert all labels to strings
        return [str(label) if not pd.isna(label) else label for label in labels]
    elif label_type == "auto":
        # Try to infer the best type
        try:
            # Check if all labels can be converted to integers
            # Skip NA values and -1 (used for missing values)
            all_int = all(
                isinstance(label, int)
                or (
                    isinstance(label, (str, float))
                    and label != -1
                    and not pd.isna(label)
                    and float(label).is_integer()
                )
                for label in labels
                if not pd.isna(label)
            )
            if all_int:
                return [
                    int(label) if label != -1 and not pd.isna(label) else label
                    for label in labels
                ]
            else:
                return [str(label) if not pd.isna(label) else label for label in labels]
        except (ValueError, TypeError):
            # If conversion fails, return as strings
            return [str(label) if not pd.isna(label) else label for label in labels]
    else:
        # Unknown label_type, return as is
        return labels


def benjamini_yekutieli_correction(
    pvals: List[float], alpha: float = 0.05
) -> List[bool]:
    """
    Applies the Benjamini–Yekutieli procedure to a list of p-values to control the false discovery rate (FDR)
    under arbitrary dependence.

    Parameters
    ----------
    pvals : List[float]
        A list of p-values from multiple hypothesis tests.
    alpha : float, optional
        The desired overall significance level (default is 0.05).

    Returns
    -------
    List[bool]
        A boolean list indicating which null hypotheses are rejected.

    Example
    -------
    >>> pvals = [0.01, 0.04, 0.03, 0.20]
    >>> benjamini_yekutieli_correction(pvals, alpha=0.05)
    [True, True, True, False]
    """
    m = len(pvals)
    sorted_indices = np.argsort(pvals)
    sorted_pvals = np.array(pvals)[sorted_indices]
    c_m = sum(1.0 / i for i in range(1, m + 1))

    rejected = [False] * m
    max_k = -1
    for k in range(m):
        threshold_k = (k + 1) * alpha / (m * c_m)
        if sorted_pvals[k] <= threshold_k:
            max_k = k
    if max_k >= 0:
        for i in range(max_k + 1):
            rejected[i] = True
    out = [False] * m
    for i, idx in enumerate(sorted_indices):
        out[idx] = rejected[i]
    return out


def accuracy_alignment(source: Any, others: List[Any]) -> float:
    """
    Computes the alignment score using the "accuracy" metric:
    Returns the average of 1.0 for each element in `others` that equals the string representation of `source`,
    and 0.0 otherwise.

    Parameters
    ----------
    source : Any
        The source value (e.g., model prediction or annotator's value).
    others : List[Any]
        A list of other annotators' values.

    Returns
    -------
    float
        The average accuracy score.

    Example
    -------
    >>> accuracy_alignment("A", ["A", "B", "A"])
    0.6666666666666666
    """
    scores = [1.0 if str(source) == str(other) else 0.0 for other in others]
    return sum(scores) / len(scores) if scores else 0.0


def rmse_alignment(source: Any, others: List[Any]) -> float:
    """
    Computes alignment for continuous values using negative RMSE (root mean squared error).
    Higher is better, so we return the negative of the RMSE.

    Parameters
    ----------
    source : Any
        The source numeric value (e.g., model prediction).
    others : List[Any]
        A list of numeric values from other annotators.

    Returns
    -------
    float
        The negative RMSE between the source and the others.
        If `others` is empty, returns 0.0 by convention.
    """
    if len(others) == 0:
        return 0.0

    # Convert source and others to float
    try:
        src_val = float(source)
    except (TypeError, ValueError):
        # If conversion fails, treat as missing => 0.0
        return 0.0

    float_others = []
    for o in others:
        try:
            float_others.append(float(o))
        except (TypeError, ValueError):
            # If conversion fails, skip or treat as 0
            float_others.append(0.0)

    # Compute RMSE
    differences = [src_val - x for x in float_others]
    mse = np.mean([d**2 for d in differences])
    rmse = np.sqrt(mse)

    # Return negative for "higher-is-better"
    return -rmse


def run_alt_test_general(
    df: pd.DataFrame,
    annotation_columns: List[str],
    model_col: str = "ModelPrediction",
    epsilon: float = 0.1,
    alpha: float = 0.05,
    metric: str = "accuracy",
    verbose: bool = True,
    label_type: str = "auto",
) -> Dict[str, Any]:
    """
    Runs the Alternative Annotator Test (alt-test) on a DataFrame containing model predictions
    and human annotations.

    For each instance and for each annotator (excluded in turn), this function computes:
        - S_llm: The alignment score between the model's prediction and the remaining annotators.
        - S_hum: The alignment score between the excluded annotator's value and the remaining annotators.

    The alignment scores are computed using one of two metrics:

    - "accuracy" (default): Uses exact match (equality) between labels.
      Uses `accuracy_alignment(source, others)`.
    - "rmse": For continuous values. Computes the negative RMSE between the source and the others.
      Uses `rmse_alignment(source, others)`.

    After computing these scores, a binary indicator is set (1.0 if S_llm > S_hum, 0.0 if S_llm < S_hum,
    and 1.0 if tied). A one-sided paired t-test is then performed on the difference (W_h - W_f) for each annotator,
    and a multiple-comparison correction (Benjamini–Yekutieli) is applied.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing model predictions and human annotations. It must include:
          - A column specified by `model_col` with the model's prediction.
          - At least three annotator columns provided via `annotation_columns`.
    annotation_columns : List[str]
        List of human annotator column names. These columns will be used for the alt-test.
        Some of these columns may be missing values per row; only valid (non-null and non-empty)
        annotations will be used. There must be at least 3 columns in this list.
    model_col : str, optional
        The column name for model predictions (default "ModelPrediction").
    epsilon : float, optional
        The cost-benefit parameter for the t-test (default 0.1).
    alpha : float, optional
        Significance level for the FDR correction (default 0.05).
    metric : str, optional
        The metric used for computing alignment scores. Options are "accuracy" (default) or "rmse".
    verbose : bool, optional
        If True, prints a summary of the alt-test results (default is True).
    label_type : str, optional
        The type to convert labels to before comparison. Options are:
        - "int": Convert all labels to integers
        - "str": Convert all labels to strings
        - "auto": Infer the best type (default)

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
            - 'annotator_columns': List of human annotator column names actually used.
            - 'pvals': List of p-values for each annotator.
            - 'rejections': Boolean list indicating which annotators the model outperforms.
            - 'winning_rate': Fraction of annotators for which the null hypothesis is rejected.
            - 'rho_f': List of advantage probabilities for the model versus each annotator.
            - 'rho_h': List of advantage probabilities for each annotator versus the model.
            - 'average_advantage_probability': The average of the model's advantage probabilities.
            - 'passed_alt_test': Boolean indicating if the model passes the alt-test (winning_rate >= 0.5).
            - 'label_counts': Dictionary with counts of valid labels for each rater.
            - 'label_types': Dictionary with types of labels for each rater.
            - 'mixed_types': Boolean indicating if there are mixed types across raters.
            - 'label_type_used': String indicating which label type conversion was used.

    Raises
    ------
    ValueError
        If there are fewer than 3 annotator columns or if the model column is missing.
    """

    if len(annotation_columns) < 3:
        raise ValueError("Need at least 3 annotator columns for the alt-test.")

    if model_col not in df.columns:
        raise ValueError(f"DataFrame does not have the model column '{model_col}'.")

    # This produces a boolean DataFrame where True means "value is not-null and not an empty string."
    valid_mask = df[annotation_columns].notna() & df[annotation_columns].ne("")

    # DEBUGGING: Count valid labels for each rater
    label_counts = {}
    label_types = {}

    # Count valid labels for the model
    model_valid = df[model_col].notna() & (df[model_col] != "")
    label_counts[model_col] = model_valid.sum()

    # Determine the type of model labels
    model_values = df.loc[model_valid, model_col].values
    model_types = set(type(val) for val in model_values if pd.notna(val))
    label_types[model_col] = [t.__name__ for t in model_types]

    # Count valid labels for each annotator
    for col in annotation_columns:
        col_valid = df[col].notna() & (df[col] != "")
        label_counts[col] = col_valid.sum()

        # Determine the type of annotator labels
        col_values = df.loc[col_valid, col].values
        col_types = set(type(val) for val in col_values if pd.notna(val))
        label_types[col] = [t.__name__ for t in col_types]

    # Check if there are mixed types across all raters
    all_types = set()
    for types in label_types.values():
        all_types.update(types)
    mixed_types = len(all_types) > 1

    if verbose:
        print("=== ALT Test: Label Debugging ===")
        print("Label counts for each rater:")
        for rater, count in label_counts.items():
            print(f"  {rater}: {count} valid labels")

        print("\nLabel types for each rater:")
        for rater, types in label_types.items():
            print(f"  {rater}: {', '.join(types)}")

        print(f"\nMixed types across raters: {mixed_types}")
        if mixed_types:
            print(f"  All types found: {', '.join(all_types)}")
        print("=" * 40)

    # Filter rows to only those with at least 3 valid annotations.
    df_valid = df[valid_mask.sum(axis=1) >= 3].copy()
    if df_valid.empty:
        raise ValueError(
            "No rows with at least 3 valid annotations; cannot perform the alt-test."
        )

    # Convert to numpy arrays.
    llm_vals_raw = df_valid[model_col].values
    # For each annotator column, get the array of values (which may contain missing entries).
    ann_arrays_raw = [df_valid[c].values for c in annotation_columns]

    # Convert labels to consistent types
    if verbose:
        print("\n=== Converting labels to consistent types ===")
        print(f"Using label_type: {label_type}")

    # Convert model predictions
    llm_vals_list = llm_vals_raw.tolist()
    llm_vals_converted = convert_labels(llm_vals_list, label_type)
    llm_vals = np.array(llm_vals_converted)

    # Convert annotator values
    ann_arrays = []
    for j, arr in enumerate(ann_arrays_raw):
        arr_list = arr.tolist()
        arr_converted = convert_labels(arr_list, label_type)
        ann_arrays.append(np.array(arr_converted))

    if verbose:
        print(
            f"Model predictions type after conversion: {type(llm_vals[0]) if len(llm_vals) > 0 else 'empty'}"
        )
        for j, col in enumerate(annotation_columns):
            val_type = (
                type(ann_arrays[j][0])
                if len(ann_arrays[j]) > 0 and not pd.isna(ann_arrays[j][0])
                else "empty/NA"
            )
            print(f"{col} type after conversion: {val_type}")

    n = len(df_valid)
    m = len(annotation_columns)

    # Validate the metric and pick the appropriate alignment function.
    if metric == "accuracy":
        S_func: Callable[[Any, List[Any]], float] = accuracy_alignment
    elif metric == "rmse":
        S_func = rmse_alignment
    else:
        raise ValueError("Unsupported metric. Use 'accuracy' or 'rmse'.")

    # Initialize lists to record wins for each annotator.
    W_f: List[List[float]] = [[] for _ in range(m)]  # LLM wins
    W_h: List[List[float]] = [[] for _ in range(m)]  # Human wins

    # Loop over each instance.
    for i in range(n):
        # For each annotator column, only compute if that annotator's value is valid.
        for j in range(m):
            # Check if the current annotator's value is valid; if not, skip this annotator for row i.
            if pd.isnull(ann_arrays[j][i]) or ann_arrays[j][i] == "":
                continue

            # For "others", include only those annotations that are valid.
            others = [
                ann_arrays[k][i]
                for k in range(m)
                if k != j and (pd.notnull(ann_arrays[k][i]) and ann_arrays[k][i] != "")
            ]
            # We require that at least 2 other annotations are present (so that overall at least 3 exist).
            if len(others) < 2:
                continue

            s_llm = S_func(llm_vals[i], others)
            s_hum = S_func(ann_arrays[j][i], others)

            if s_llm > s_hum:
                W_f[j].append(1.0)
                W_h[j].append(0.0)
            elif s_llm < s_hum:
                W_f[j].append(0.0)
                W_h[j].append(1.0)
            else:
                # In case of a tie, count both as wins.
                W_f[j].append(1.0)
                W_h[j].append(1.0)

    # Compute advantage probabilities for each annotator.
    rho_f_vals = [np.mean(wins) if wins else np.nan for wins in W_f]
    rho_h_vals = [np.mean(wins) if wins else np.nan for wins in W_h]

    # For each annotator, perform a one-sided t-test on the difference (W_h - W_f) versus epsilon.
    pvals = []
    for j in range(m):
        # Only perform the t-test if we have any samples for this annotator.
        if len(W_f[j]) == 0:
            pvals.append(np.nan)
            continue
        d_j = np.array(W_h[j]) - np.array(W_f[j])
        # t-test: alternative='less' tests H0: mean(d_j) >= epsilon vs H1: mean(d_j) < epsilon.
        _, p_val = ttest_1samp(d_j, popmean=epsilon, alternative="less")
        pvals.append(p_val)

    # Apply multiple-comparison correction (Benjamini–Yekutieli).
    rejections = benjamini_yekutieli_correction(pvals, alpha=alpha)
    winning_rate = np.nanmean(np.array(rejections))
    avg_adv_prob = np.nanmean(np.array(rho_f_vals))
    passed_alt_test = winning_rate >= 0.5

    if verbose:
        print("=== Alt-Test: summary ===")
        print("P-values for each comparison:")
        for j in range(m):
            print(
                f"{annotation_columns[j]}: p={pvals[j]:.4f} => rejectH0={rejections[j]} | "
                f"rho_f={rho_f_vals[j]:.3f}, rho_h={rho_h_vals[j]:.3f}"
            )
        print("\nSummary statistics:")
        print(f"Winning Rate (omega) = {winning_rate:.3f}")
        print(f"Average Advantage Probability (rho) = {avg_adv_prob:.3f}")
        print(f"Passed Alt-Test? => {passed_alt_test}")

    return {
        "annotator_columns": annotation_columns,
        "pvals": pvals,
        "rejections": rejections,
        "winning_rate": winning_rate,
        "rho_f": rho_f_vals,
        "rho_h": rho_h_vals,
        "average_advantage_probability": avg_adv_prob,
        "passed_alt_test": passed_alt_test,
        "label_counts": label_counts,
        "label_types": label_types,
        "mixed_types": mixed_types,
        "label_type_used": label_type,
    }
