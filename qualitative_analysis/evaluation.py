"""
evaluation.py

This module provides functions for computing Cohen's Kappa scores and visualizing 
confusion matrices to assess classification accuracy and inter-rater reliability 
between model predictions and human annotations.

Dependencies:
    - scikit-learn
    - matplotlib
    - seaborn

Functions:
    - compute_cohens_kappa(judgments_1, judgments_2, labels=None, weights=None): 
      Calculates Cohen's Kappa score between two sets of categorical labels.

    - compute_all_kappas(model_coding, human_annotations, labels=None, weights=None, verbose=False): 
      Computes Cohen's Kappa scores between model predictions and multiple human annotators, 
      as well as between human annotators themselves.

    - plot_confusion_matrices(model_coding, human_annotations, labels): 
      Generates confusion matrices comparing model predictions to human annotations 
      and between human annotators, visualized as heatmaps.
"""

from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Optional, Union, Mapping


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
) -> dict[str, float]:
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


def plot_confusion_matrices(
    model_coding: List[Union[int, str]],
    human_annotations: Dict[str, List[Union[int, str]]],
    labels: List[Union[int, str]],
) -> None:
    """
    Plots confusion matrices comparing model predictions to human annotations
    and between human annotators themselves.

    Two types of confusion matrices are generated:
        1. **Model vs. Human Raters:** Visual comparison between model predictions and each human rater.
        2. **Human vs. Human Raters:** Pairwise comparison between human annotators.

    Parameters:
    ----------
    model_coding : List[int] or List[str]
        List of model predictions for each sample.

    human_annotations : Dict[str, List[int] or List[str]]
        Dictionary where keys are rater names and values are lists of annotations.

    labels : List[int] or List[str]
        List of unique labels to index the confusion matrix.

    Returns:
    -------
    None
        Displays confusion matrix heatmaps for comparison.

    Raises:
    ------
    ValueError
        If the lengths of model predictions and human annotations do not match.
    """
    sns.set_theme(style="whitegrid")

    raters = list(human_annotations.keys())
    n_raters = len(raters)

    # Check for length mismatches
    for rater, annotations in human_annotations.items():
        if len(model_coding) != len(annotations):
            raise ValueError(
                f"Length mismatch between model predictions and {rater}'s annotations."
            )

    # === Model vs. Human Raters ===
    fig, axes = plt.subplots(1, n_raters, figsize=(6 * n_raters, 5))
    if n_raters == 1:
        axes = [axes]

    for idx, (rater, annotations) in enumerate(human_annotations.items()):
        cm = confusion_matrix(annotations, model_coding, labels=labels)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[idx],
        )
        axes[idx].set_title(f"Model vs {rater}")
        axes[idx].set_xlabel("Model Predictions")
        axes[idx].set_ylabel(f"{rater} Annotations")

    plt.tight_layout()
    plt.show()

    # === Human Rater vs. Human Rater ===
    if n_raters > 1:
        num_comparisons = n_raters * (n_raters - 1) // 2
        fig, axes = plt.subplots(1, num_comparisons, figsize=(6 * num_comparisons, 5))
        if num_comparisons == 1:
            axes = [axes]

        idx = 0
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                rater1, rater2 = raters[i], raters[j]
                cm = confusion_matrix(
                    human_annotations[rater1], human_annotations[rater2], labels=labels
                )
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels,
                    ax=axes[idx],
                )
                axes[idx].set_title(f"{rater1} vs {rater2}")
                axes[idx].set_xlabel(f"{rater2} Annotations")
                axes[idx].set_ylabel(f"{rater1} Annotations")
                idx += 1

        plt.tight_layout()
        plt.show()
