"""
kappa.py

This module provides functions for computing Cohen's Kappa scores, Krippendorff's alpha,
and related metrics to assess inter-rater reliability between model predictions and human annotations.

Functions:
    - compute_cohens_kappa(judgments_1, judgments_2, labels=None, weights=None): 
      Calculates Cohen's Kappa score between two sets of categorical labels.

    - compute_all_kappas(model_coding, human_annotations, labels=None, weights=None, verbose=False): 
      Computes Cohen's Kappa scores between model predictions and multiple human annotators, 
      as well as between human annotators themselves.

    - compute_detailed_kappa_metrics(model_predictions, human_annotations, labels=None, kappa_weights=None):
      Compute detailed kappa metrics between model and human annotators, and between human annotators.

    - compute_kappa_metrics(detailed_results_df, annotation_columns, labels, kappa_weights=None):
      Compute kappa metrics from detailed results DataFrame.

    - compute_krippendorff_non_inferiority(detailed_results_df, annotation_columns, model_column, 
      level_of_measurement, non_inferiority_margin, n_bootstrap, confidence_level, random_seed, verbose):
      Test if model annotations are non-inferior to human annotations using Krippendorff's alpha 
      with configurable confidence intervals.

    - print_non_inferiority_results(non_inferiority_results, show_per_run):
      Print non-inferiority test results in a formatted way.
"""

from sklearn.metrics import cohen_kappa_score, accuracy_score
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Mapping, Any, Tuple
import krippendorff

# Import utility functions
from qualitative_analysis.metrics.utils import (
    compute_majority_vote,
    ensure_numeric_columns,
)


def compute_cohens_kappa(
    judgments_1: Union[List[int], List[str]],
    judgments_2: Union[List[int], List[str]],
    labels: Optional[List[Union[int, str]]] = None,
    weights: Optional[str] = None,
) -> float:
    """
    Computes Cohen's Kappa score between two sets of categorical judgments.

    Parameters:
    ----------
    judgments_1 : List[int] or List[str]
        Labels assigned by the first rater.
    judgments_2 : List[int] or List[str]
        Labels assigned by the second rater.
    labels : List[int] or List[str], optional
        List of unique labels to index the confusion matrix. If not provided,
        the union of labels from both raters is used.
    weights : str, optional
        Weighting scheme for computing Kappa. Options are:
            - `'linear'`: Penalizes disagreements linearly.
            - `'quadratic'`: Penalizes disagreements quadratically.
            - `None`: No weighting (default).

    Returns:
    -------
    float
        Cohen's Kappa score, ranging from:
            - `1.0`: Perfect agreement.
            - `0.0`: Agreement equal to chance.
            - `-1.0`: Complete disagreement.

    Raises:
    ------
    ValueError
        If input lists have different lengths or contain invalid labels.

    Examples:
    --------
    Basic usage without specifying labels or weights:

    >>> judgments_1 = [0, 1, 2, 1, 0]
    >>> judgments_2 = [0, 2, 2, 1, 0]
    >>> round(compute_cohens_kappa(judgments_1, judgments_2), 2)
    0.71

    Using specified labels and linear weights:

    >>> labels = [0, 1, 2]
    >>> round(compute_cohens_kappa(judgments_1, judgments_2, labels=labels, weights='linear'), 2)
    0.78

    Handling perfect agreement:

    >>> judgments_3 = [0, 1, 2, 1, 0]
    >>> round(compute_cohens_kappa(judgments_1, judgments_3), 2)
    1.0

    Example with no agreement beyond chance:

    >>> judgments_4 = [2, 0, 1, 2, 1]
    >>> round(compute_cohens_kappa(judgments_1, judgments_4), 2)
    -0.47

    References:
    ----------
    - Cohen, J. (1960). A coefficient of agreement for nominal scales.
      *Educational and Psychological Measurement*, 20(1), 37–46.
    """
    return cohen_kappa_score(judgments_1, judgments_2, labels=labels, weights=weights)


def compute_all_kappas(
    model_coding: Union[List[int], List[str]],
    human_annotations: Mapping[str, Union[List[int], List[str]]],
    labels: Optional[List[Union[int, str]]] = None,
    weights: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Computes Cohen's Kappa scores for all combinations of model predictions
    and human annotations, as well as between human annotations themselves.

    Parameters:
    ----------
    model_coding : List[int] or List[str]
        Model predictions for each sample.

    human_annotations : Dict[str, List[int] or List[str]]
        Dictionary where keys are rater names and values are lists of annotations.

    labels : List[int] or List[str], optional
        List of unique labels to index the confusion matrix.
        If not provided, the union of all labels is used.

    weights : str, optional
        Weighting scheme for computing Kappa. Options are:
            - `'linear'`: Linear penalty for disagreements.
            - `'quadratic'`: Quadratic penalty for disagreements.
            - `None`: No weighting (default).

    verbose : bool, optional
        If `True`, prints the Kappa scores during computation. Default is `False`.

    Returns:
    -------
    Dict[str, float]
        A dictionary containing Cohen's Kappa scores for all comparisons:
            - `"model_vs_<rater>"`: Kappa between the model and each human rater.
            - `"<rater1>_vs_<rater2>"`: Kappa between each pair of human raters.

    Raises:
    ------
    ValueError
        If the annotations have inconsistent lengths.

    Examples:
    -------
    >>> model_coding = [0, 1, 2, 1, 0]
    >>> human_annotations = {
    ...     "Rater1": [0, 1, 2, 0, 0],
    ...     "Rater2": [0, 2, 2, 1, 0]
    ... }
    >>> labels = [0, 1, 2]
    >>> result = compute_all_kappas(model_coding, human_annotations, labels=labels)
    >>> sorted(result.keys())
    ['Rater1_vs_Rater2', 'model_vs_Rater1', 'model_vs_Rater2']

    >>> invalid_annotations = {
    ...     "Rater1": [0, 1],  # Mismatched length
    ...     "Rater2": [0, 2, 2, 1, 0]
    ... }
    >>> compute_all_kappas(model_coding, invalid_annotations, labels=labels)
    Traceback (most recent call last):
        ...
    ValueError: Length mismatch: model_coding and Rater1's annotations must have the same length.
    """
    results = {}

    # Compare model with each human rater
    for rater, annotations in human_annotations.items():
        if len(model_coding) != len(annotations):
            raise ValueError(
                f"Length mismatch: model_coding and {rater}'s annotations must have the same length."
            )

        kappa = compute_cohens_kappa(
            model_coding, annotations, labels=labels, weights=weights
        )
        results[f"model_vs_{rater}"] = kappa
        if verbose:
            print(f"model vs {rater}: {kappa:.2f}")

    # Compare each human rater with every other human rater
    raters = list(human_annotations.keys())
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            rater1, rater2 = raters[i], raters[j]

            if len(human_annotations[rater1]) != len(human_annotations[rater2]):
                raise ValueError(
                    f"Length mismatch: {rater1} and {rater2}'s annotations must have the same length."
                )

            kappa = compute_cohens_kappa(
                human_annotations[rater1],
                human_annotations[rater2],
                labels=labels,
                weights=weights,
            )
            results[f"{rater1}_vs_{rater2}"] = kappa
            if verbose:
                print(f"{rater1} vs {rater2}: {kappa:.2f}")

    return results


def compute_detailed_kappa_metrics(
    model_predictions: List[Any],
    human_annotations: Dict[str, List[Any]],
    labels: Optional[List[Any]] = None,
    kappa_weights: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute detailed kappa metrics between model and human annotators, and between human annotators.

    Parameters
    ----------
    model_predictions : List[Any]
        List of model predictions.
    human_annotations : Dict[str, List[Any]]
        Dictionary mapping annotator names to lists of annotations.
    labels : Optional[List[Any]], optional
        List of valid labels, by default None
    kappa_weights : Optional[str], optional
        Weighting scheme for kappa calculation ('linear', 'quadratic', or None), by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary containing detailed kappa metrics:
        - 'mean_llm_human_agreement': Mean kappa between LLM and human annotators
        - 'mean_human_human_agreement': Mean kappa between human annotators
        - 'llm_vs_human_df': DataFrame with kappa between LLM and each human annotator
        - 'human_vs_human_df': DataFrame with kappa between each pair of human annotators
    """
    # Compute kappa between model and each human annotator
    llm_human_kappas = []
    llm_human_data = []

    for annotator, annotations in human_annotations.items():
        kappa = compute_cohens_kappa(
            model_predictions, annotations, labels=labels, weights=kappa_weights
        )
        llm_human_kappas.append(kappa)
        llm_human_data.append({"Human_Annotator": annotator, "Cohens_Kappa": kappa})

    # Compute mean LLM-Human agreement
    mean_llm_human_agreement = np.mean(llm_human_kappas) if llm_human_kappas else None

    # Create DataFrame for LLM vs Human kappas
    llm_human_df = pd.DataFrame(llm_human_data)

    # Compute kappa between each pair of human annotators
    human_human_kappas = []
    human_human_data = []

    annotators = list(human_annotations.keys())
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            annotator1 = annotators[i]
            annotator2 = annotators[j]

            kappa = compute_cohens_kappa(
                human_annotations[annotator1],
                human_annotations[annotator2],
                labels=labels,
                weights=kappa_weights,
            )
            human_human_kappas.append(kappa)
            human_human_data.append(
                {
                    "Annotator_1": annotator1,
                    "Annotator_2": annotator2,
                    "Cohens_Kappa": kappa,
                }
            )

    # Compute mean Human-Human agreement
    mean_human_human_agreement = (
        np.mean(human_human_kappas) if human_human_kappas else None
    )

    # Create DataFrame for Human vs Human kappas
    human_human_df = pd.DataFrame(human_human_data)

    return {
        "mean_llm_human_agreement": mean_llm_human_agreement,
        "mean_human_human_agreement": mean_human_human_agreement,
        "llm_vs_human_df": llm_human_df,
        "human_vs_human_df": human_human_df,
    }


def compute_krippendorff_non_inferiority(
    detailed_results_df: pd.DataFrame,
    annotation_columns: List[str],
    model_column: str = "ModelPrediction",
    level_of_measurement: str = "ordinal",
    non_inferiority_margin: float = -0.05,
    n_bootstrap: int = 2000,
    confidence_level: float = 90.0,
    random_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Test if model annotations are non-inferior to human annotations using Krippendorff's alpha.

    This function implements a non-inferiority test comparing trios of human annotators to
    trios that include the model. It uses bootstrap resampling to compute confidence intervals
    and determine if the model is non-inferior to human annotators within a specified margin.

    Parameters
    ----------
    detailed_results_df : pd.DataFrame
        DataFrame containing detailed results
    annotation_columns : List[str]
        List of column names containing human annotations
    model_column : str, optional
        Column name containing model predictions, by default "ModelPrediction"
    level_of_measurement : str, optional
        Level of measurement for Krippendorff's alpha, by default 'ordinal'
    non_inferiority_margin : float, optional
        Non-inferiority margin (delta), by default -0.05
    n_bootstrap : int, optional
        Number of bootstrap samples, by default 2000
    confidence_level : float, optional
        Confidence level for the confidence interval (e.g., 90.0 for 90% CI), by default 90.0
    random_seed : int, optional
        Random seed for reproducibility, by default 42
    verbose : bool, optional
        Whether to print detailed results, by default True

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary containing test results for each scenario
    """
    import itertools

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_seed)

    # Ensure numeric columns
    columns_to_check = annotation_columns + [model_column]
    detailed_results_df = ensure_numeric_columns(detailed_results_df, columns_to_check)

    # Group by scenario (prompt_name, iteration)
    scenario_grouped = detailed_results_df.groupby(["prompt_name", "iteration"])

    # Store results for each scenario
    scenario_results = {}

    # Helper function to compute alpha for a subset of columns
    def alpha_cols(frame, cols, row_idx=None):
        """Compute Krippendorff's alpha for specified columns and rows."""
        sub = frame.iloc[row_idx] if row_idx is not None else frame
        data = sub[cols].dropna().T.to_numpy(dtype=float)
        return krippendorff.alpha(data, level_of_measurement=level_of_measurement)

    for (prompt_name, iteration), scenario_group in scenario_grouped:
        scenario_key = f"{prompt_name}_iteration_{iteration}"

        if verbose:
            print(f"\n=== Non-inferiority Test: {scenario_key} ===")

        # Process each run separately
        run_results = []

        for run_id in sorted(scenario_group["run"].unique()):
            run_data = scenario_group[scenario_group["run"] == run_id]

            # Split into train and validation
            train_data = run_data[run_data["split"] == "train"]

            # Skip if no training data
            if len(train_data) == 0:
                continue

            try:
                # Define the groups to compare
                human_trios = list(itertools.combinations(annotation_columns, 3))
                model_trios = [
                    list(t)
                    for t in itertools.combinations(
                        annotation_columns + [model_column], 3
                    )
                    if model_column in t
                ]

                # If there are fewer than 3 human annotators, we can't form human-only trios
                if len(human_trios) == 0:
                    if verbose:
                        print(
                            f"  Run {run_id}: Skipping - need at least 3 human annotators"
                        )
                    continue

                # Compute alpha for human-only trios
                alpha_human_trios = [
                    alpha_cols(train_data, list(trio)) for trio in human_trios
                ]
                alpha_H = np.mean(alpha_human_trios)

                # Compute alpha for model-included trios
                alpha_model_trios = [
                    alpha_cols(train_data, trio) for trio in model_trios
                ]
                alpha_M = np.mean(alpha_model_trios)

                # Observed difference
                observed_difference = alpha_M - alpha_H

                # Bootstrap test
                n_items = len(train_data)
                deltas = []

                for _ in range(n_bootstrap):
                    # Sample with replacement
                    idx = rng.integers(0, n_items, n_items)

                    # Compute alphas for bootstrap samples
                    alpha_H_b = np.mean(
                        [
                            alpha_cols(train_data, list(trio), idx)
                            for trio in human_trios
                        ]
                    )
                    alpha_M_b = np.mean(
                        [alpha_cols(train_data, trio, idx) for trio in model_trios]
                    )

                    # Store difference
                    deltas.append(alpha_M_b - alpha_H_b)

                # Compute confidence interval based on confidence_level
                alpha_level = (100 - confidence_level) / 2
                ci_low = np.percentile(deltas, alpha_level)
                ci_high = np.percentile(deltas, 100 - alpha_level)

                # Determine if non-inferiority is demonstrated
                non_inferiority = ci_low > non_inferiority_margin

                # Store results for this run
                run_result = {
                    "run": run_id,
                    "alpha_human_trios": alpha_H,
                    "alpha_model_trios": alpha_M,
                    "difference": observed_difference,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "non_inferiority_margin": non_inferiority_margin,
                    "non_inferiority_demonstrated": non_inferiority,
                    "n_bootstrap_samples": n_bootstrap,
                    "human_trios": human_trios,
                    "model_trios": model_trios,
                    "bootstrap_differences": np.array(deltas),
                }

                run_results.append(run_result)

                if verbose:
                    print(f"\n  Run {run_id}:")
                    print(f"    Human trios α: {alpha_H:.4f}")
                    print(f"    Model trios α: {alpha_M:.4f}")
                    print(f"    Δ = model − human = {observed_difference:+.4f}")
                    print(
                        f"    {confidence_level:.0f}% CI: [{ci_low:.4f}, {ci_high:.4f}]"
                    )
                    if non_inferiority:
                        print(
                            f"    ✅ Non-inferiority demonstrated (margin = {non_inferiority_margin})"
                        )
                    else:
                        print(
                            f"    ❌ Non-inferiority NOT demonstrated (margin = {non_inferiority_margin})"
                        )

            except Exception as e:
                if verbose:
                    print(f"  Run {run_id}: Error - {str(e)}")

        # Aggregate results across runs
        if run_results:
            # Calculate means and standard deviations
            mean_alpha_H = np.mean([r["alpha_human_trios"] for r in run_results])
            std_alpha_H = np.std([r["alpha_human_trios"] for r in run_results])

            mean_alpha_M = np.mean([r["alpha_model_trios"] for r in run_results])
            std_alpha_M = np.std([r["alpha_model_trios"] for r in run_results])

            mean_difference = np.mean([r["difference"] for r in run_results])
            std_difference = np.std([r["difference"] for r in run_results])

            mean_ci_low = np.mean([r["ci_lower"] for r in run_results])
            mean_ci_high = np.mean([r["ci_upper"] for r in run_results])

            # Count non-inferiority demonstrations
            n_non_inferior = sum(
                [r["non_inferiority_demonstrated"] for r in run_results]
            )

            # Store aggregated results
            aggregated_metrics = {
                "n_runs": len(run_results),
                "alpha_human_trios_mean": mean_alpha_H,
                "alpha_human_trios_std": std_alpha_H,
                "alpha_model_trios_mean": mean_alpha_M,
                "alpha_model_trios_std": std_alpha_M,
                "difference_mean": mean_difference,
                "difference_std": std_difference,
                "ci_lower_mean": mean_ci_low,
                "ci_upper_mean": mean_ci_high,
                "confidence_level": confidence_level,
                "non_inferiority_margin": non_inferiority_margin,
                "n_non_inferior": n_non_inferior,
                "non_inferiority_ratio": n_non_inferior / len(run_results),
            }

            # Store all results for this scenario
            scenario_results[scenario_key] = {
                "run_results": run_results,
                "aggregated_metrics": aggregated_metrics,
            }

            # Print summary
            if verbose:
                print(f"\n  Summary across {len(run_results)} runs:")
                print(f"    Human trios α: {mean_alpha_H:.4f} ± {std_alpha_H:.4f}")
                print(f"    Model trios α: {mean_alpha_M:.4f} ± {std_alpha_M:.4f}")
                print(
                    f"    Δ = model − human = {mean_difference:+.4f} ± {std_difference:.4f}"
                )
                print(
                    f"    {confidence_level:.0f}% CI: [{mean_ci_low:.4f}, {mean_ci_high:.4f}]"
                )
                print(
                    f"    Non-inferiority demonstrated in {n_non_inferior}/{len(run_results)} runs"
                )

                if n_non_inferior == len(run_results):
                    print(
                        f"    ✅ Non-inferiority consistently demonstrated across all runs (margin = {non_inferiority_margin})"
                    )
                elif n_non_inferior > 0:
                    print(
                        f"    ⚠️ Non-inferiority demonstrated in some but not all runs (margin = {non_inferiority_margin})"
                    )
                else:
                    print(
                        f"    ❌ Non-inferiority NOT demonstrated in any run (margin = {non_inferiority_margin})"
                    )

    return scenario_results


def print_non_inferiority_results(
    non_inferiority_results: Dict[str, Dict[str, Any]], show_per_run: bool = False
) -> None:
    """
    Print non-inferiority test results in a formatted way.

    Parameters
    ----------
    non_inferiority_results : Dict[str, Dict[str, Any]]
        Results from compute_krippendorff_non_inferiority
    show_per_run : bool, optional
        Whether to show individual run details, by default False
    """
    for scenario_key, results in non_inferiority_results.items():
        print(f"\n=== Non-inferiority Test: {scenario_key} ===")

        # Access the aggregated metrics
        agg = results["aggregated_metrics"]

        print(
            f"Human trios α: {agg['alpha_human_trios_mean']:.4f} ± {agg['alpha_human_trios_std']:.4f}"
        )
        print(
            f"Model trios α: {agg['alpha_model_trios_mean']:.4f} ± {agg['alpha_model_trios_std']:.4f}"
        )
        print(
            f"Δ = model − human = {agg['difference_mean']:+.4f} ± {agg['difference_std']:.4f}"
        )
        print(
            f"{agg['confidence_level']:.0f}% CI: [{agg['ci_lower_mean']:.4f}, {agg['ci_upper_mean']:.4f}]"
        )
        print(
            f"Non-inferiority demonstrated in {agg['n_non_inferior']}/{agg['n_runs']} runs"
        )

        if agg["n_non_inferior"] == agg["n_runs"]:
            print(
                f"✅ Non-inferiority consistently demonstrated across all runs (margin = {agg['non_inferiority_margin']})"
            )
        elif agg["n_non_inferior"] > 0:
            print(
                f"⚠️ Non-inferiority demonstrated in some but not all runs (margin = {agg['non_inferiority_margin']})"
            )
        else:
            print(
                f"❌ Non-inferiority NOT demonstrated in any run (margin = {agg['non_inferiority_margin']})"
            )

        # Show individual run details if requested
        if show_per_run:
            print("\nDetailed per-run results:")
            for run_result in results["run_results"]:
                run_id = run_result["run"]
                confidence_level = agg.get("confidence_level", 90.0)
                print(f"  Run {run_id}:")
                print(f"    Human trios α: {run_result['alpha_human_trios']:.4f}")
                print(f"    Model trios α: {run_result['alpha_model_trios']:.4f}")
                print(f"    Δ = model − human = {run_result['difference']:+.4f}")
                print(
                    f"    {confidence_level:.0f}% CI: [{run_result['ci_lower']:.4f}, {run_result['ci_upper']:.4f}]"
                )
                if run_result["non_inferiority_demonstrated"]:
                    print(
                        f"    ✅ Non-inferiority demonstrated (margin = {run_result['non_inferiority_margin']})"
                    )
                else:
                    print(
                        f"    ❌ Non-inferiority NOT demonstrated (margin = {run_result['non_inferiority_margin']})"
                    )


def compute_kappa_metrics(
    detailed_results_df: pd.DataFrame,
    annotation_columns: List[str],
    labels: List[Any],
    kappa_weights: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Compute kappa metrics from detailed results DataFrame.

    Parameters
    ----------
    detailed_results_df : pd.DataFrame
        DataFrame containing detailed results from run_scenarios.
    annotation_columns : List[str]
        List of column names containing human annotations.
    labels : List[Any]
        List of valid labels.
    kappa_weights : Optional[str], optional
        Weighting scheme for kappa calculation ('linear', 'quadratic', or None), by default None

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]
        A tuple containing:
        - A DataFrame with aggregated kappa metrics
        - A dictionary containing detailed kappa metrics for each scenario
    """
    # For the final summary dataframe
    all_aggregated_results = []

    # For storing detailed kappa metrics
    detailed_kappa_metrics = {}

    # Print the columns in the detailed_results_df to debug
    print("\n=== Columns in detailed_results_df (in compute_kappa_metrics) ===")
    print(detailed_results_df.columns.tolist())

    # Check if the required columns are present in the detailed_results_df
    required_columns = [
        "prompt_name",
        "iteration",
        "ModelPrediction",
    ] + annotation_columns
    missing_columns = [
        col for col in required_columns if col not in detailed_results_df.columns
    ]

    if missing_columns:
        print(
            f"Warning: The following required columns are missing from the detailed results DataFrame: {missing_columns}"
        )
        print("Kappa metrics will not be computed.")
        # Return empty DataFrames
        return pd.DataFrame(), {}

    # Ensure numeric columns
    detailed_results_df = ensure_numeric_columns(
        detailed_results_df, ["ModelPrediction"] + annotation_columns
    )

    # Group by scenario, prompt_name, and iteration
    grouped = detailed_results_df.groupby(["prompt_name", "iteration"])

    for (prompt_name, iteration), group in grouped:
        # Split data into train and validation sets
        train_data = group[group["split"] == "train"]
        val_data = group[group["split"] == "val"]

        # Extract validation setting
        use_validation_set = len(val_data) > 0

        # Extract model predictions and human annotations for train data
        train_model_predictions = train_data["ModelPrediction"].tolist()
        train_human_annotations = {
            col: train_data[col].tolist() for col in annotation_columns
        }

        # Extract model predictions and human annotations for validation data if available
        val_model_predictions = (
            val_data["ModelPrediction"].tolist() if use_validation_set else []
        )
        val_human_annotations = (
            {col: val_data[col].tolist() for col in annotation_columns}
            if use_validation_set
            else {}
        )

        # Initialize aggregated metrics
        aggregated_metrics = {
            "prompt_name": prompt_name,
            "iteration": iteration,
            "n_runs": len(set(group["run"])),
            "use_validation_set": use_validation_set,
            "N_train": len(train_data),
            "N_val": len(val_data) if use_validation_set else 0,
        }

        # Compute metrics for train data
        if train_model_predictions and all(
            len(annotations) > 0 for annotations in train_human_annotations.values()
        ):
            try:
                # Compute accuracy and kappa for train data
                train_ground_truth = compute_majority_vote(train_human_annotations)
                accuracy_train = accuracy_score(
                    train_ground_truth, train_model_predictions
                )
                kappa_train = compute_cohens_kappa(
                    train_ground_truth,
                    train_model_predictions,
                    labels=labels,
                    weights=kappa_weights,
                )

                # Add basic metrics to aggregated_metrics
                aggregated_metrics["accuracy_train"] = accuracy_train
                aggregated_metrics["kappa_train"] = kappa_train

                # Compute detailed kappa metrics for train data
                train_kappa_metrics = compute_detailed_kappa_metrics(
                    model_predictions=train_model_predictions,
                    human_annotations=train_human_annotations,
                    labels=labels,
                    kappa_weights=kappa_weights,
                )

                # Add mean agreement scores to aggregated metrics
                aggregated_metrics["mean_llm_human_agreement"] = train_kappa_metrics[
                    "mean_llm_human_agreement"
                ]
                aggregated_metrics["mean_human_human_agreement"] = train_kappa_metrics[
                    "mean_human_human_agreement"
                ]

                # Store DataFrames for detailed reporting
                scenario_key = f"{prompt_name}_iteration_{iteration}"
                detailed_kappa_metrics[scenario_key] = {
                    "llm_vs_human_df": train_kappa_metrics["llm_vs_human_df"],
                    "human_vs_human_df": train_kappa_metrics["human_vs_human_df"],
                }

            except Exception as e:
                print(f"Error computing train kappa metrics: {e}")

        # Compute metrics for validation data if available
        if (
            use_validation_set
            and val_model_predictions
            and all(
                len(annotations) > 0 for annotations in val_human_annotations.values()
            )
        ):
            try:
                # Compute accuracy and kappa for validation data
                val_ground_truth = compute_majority_vote(val_human_annotations)
                accuracy_val = accuracy_score(val_ground_truth, val_model_predictions)
                kappa_val = compute_cohens_kappa(
                    val_ground_truth,
                    val_model_predictions,
                    labels=labels,
                    weights=kappa_weights,
                )

                # Add basic metrics to aggregated_metrics
                aggregated_metrics["accuracy_val"] = accuracy_val
                aggregated_metrics["kappa_val"] = kappa_val

                # Compute detailed kappa metrics for validation data
                val_kappa_metrics = compute_detailed_kappa_metrics(
                    model_predictions=val_model_predictions,
                    human_annotations=val_human_annotations,
                    labels=labels,
                    kappa_weights=kappa_weights,
                )

                # Add mean agreement scores to aggregated metrics
                aggregated_metrics["mean_llm_human_agreement_val"] = val_kappa_metrics[
                    "mean_llm_human_agreement"
                ]
                aggregated_metrics["mean_human_human_agreement_val"] = (
                    val_kappa_metrics["mean_human_human_agreement"]
                )

            except Exception as e:
                print(f"Error computing validation kappa metrics: {e}")

        # Add to the list of aggregated results
        all_aggregated_results.append(aggregated_metrics)

    # Create DataFrame from the results
    summary_df = pd.DataFrame(all_aggregated_results)

    return summary_df, detailed_kappa_metrics
