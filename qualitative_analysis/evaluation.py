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
      
    - compute_classification_metrics(model_coding, human_annotations, labels=None): 
      Computes detailed classification metrics including accuracy, recall, and error rates
      globally and per class, using majority vote of human annotations as ground truth.
"""

from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    confusion_matrix,
    recall_score,
)
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Mapping, Any, TypedDict, cast
import pandas as pd


# TypedDict definitions for classification metrics
class ClassMetrics(TypedDict):
    recall: float
    error_rate: float
    correct_count: int  # TP
    missed_count: int  # FN
    false_positives: int  # FP


class GlobalMetrics(TypedDict):
    accuracy: float
    recall: float
    error_rate: float


class ClassificationResults(TypedDict):
    class_distribution: Dict[Union[int, str], int]
    global_metrics: Dict[str, GlobalMetrics]
    per_class_metrics: Dict[Union[int, str], Dict[str, ClassMetrics]]
    ground_truth: List[Any]
    confusion_matrices: Dict[str, np.ndarray]


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
    human_annotations: Dict[str, List[int]],
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

    human_annotations : Dict[str, List[int]]
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


def compute_human_accuracies(df, annotation_columns, ground_truth_column="GroundTruth"):
    """
    Computes the accuracy for each human annotator using sklearn's accuracy_score.

    Parameters:
      df (pd.DataFrame): DataFrame containing the annotator columns and ground truth.
      annotation_columns (List[str]): List of human annotator column names.
      ground_truth_column (str): Name of the column with the ground truth labels.

    Returns:
      Dict[str, float]: A dictionary mapping each annotator to their accuracy.
    """
    accuracies = {}
    for col in annotation_columns:
        # Filter out invalid annotations.
        valid_mask = df[col].notnull() & (df[col] != "")
        if valid_mask.sum() > 0:
            y_true = df.loc[valid_mask, ground_truth_column]
            y_pred = df.loc[valid_mask, col]
            accuracies[col] = accuracy_score(y_true, y_pred)
        else:
            accuracies[col] = float("nan")
    return accuracies


def compute_majority_vote(annotations: Dict[str, List], ignore_na: bool = True) -> List:
    """
    Computes the majority vote for each instance across multiple annotators.

    Parameters:
    ----------
    annotations : Dict[str, List]
        Dictionary where keys are annotator names and values are lists of annotations.
    ignore_na : bool, optional
        If True, ignores NA values when computing the majority vote. Default is True.

    Returns:
    -------
    List
        A list containing the majority vote for each instance.
    """
    if not annotations:
        return []

    # Get the number of instances
    n_instances = len(next(iter(annotations.values())))

    # Check that all annotators have the same number of instances
    for annotator, labels in annotations.items():
        if len(labels) != n_instances:
            raise ValueError(
                f"Annotator {annotator} has {len(labels)} instances, expected {n_instances}"
            )

    # Compute majority vote for each instance
    majority_votes = []
    for i in range(n_instances):
        instance_labels = []
        for annotator, labels in annotations.items():
            label = labels[i]
            if ignore_na and (pd.isna(label) or label == ""):
                continue
            instance_labels.append(label)

        if not instance_labels:
            # If all annotations are NA, use NA as the majority vote
            majority_votes.append(np.nan)
        else:
            # Count occurrences of each label
            label_counts = Counter(instance_labels)
            # Find the label with the highest count
            majority_label, _ = label_counts.most_common(1)[0]
            majority_votes.append(majority_label)

    return majority_votes


def compute_classification_metrics(
    model_coding: Union[List[int], List[str]],
    human_annotations: Dict[str, Union[List[int], List[str]]],
    labels: Optional[List[Union[int, str]]] = None,
) -> ClassificationResults:
    """
    Computes detailed classification metrics including accuracy, recall, and error rates
    globally and per class, using majority vote of human annotations as ground truth.

    Parameters:
    ----------
    model_coding : List[int] or List[str]
        Model predictions for each sample.

    human_annotations : Dict[str, List[int] or List[str]]
        Dictionary where keys are annotator names and values are lists of annotations.

    labels : List[int] or List[str], optional
        List of unique labels to use. If not provided, the unique values from
        the model predictions and human annotations will be used.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing:
            - 'class_distribution': Count of instances for each class in the ground truth
            - 'global_metrics': Overall metrics for the model and each human annotator
                - 'accuracy': Proportion of correct predictions
                - 'error_rate': Proportion of incorrect predictions (1 - accuracy)
                - 'recall': Average recall across all classes
            - 'per_class_metrics': Metrics for each class
                - 'recall': Proportion of actual positives correctly identified
                - 'error_rate': Proportion of instances of this class that were misclassified
                - 'correct_count': Number of correctly identified instances
                - 'missed_count': Number of instances that were misclassified
            - 'ground_truth': The majority vote used as ground truth
            - 'confusion_matrices': Confusion matrices for the model and each human annotator

    Raises:
    ------
    ValueError
        If the annotations have inconsistent lengths.
    """
    # Check for length mismatches
    for rater, annotations in human_annotations.items():
        if len(model_coding) != len(annotations):
            raise ValueError(
                f"Length mismatch: model_coding and {rater}'s annotations must have the same length."
            )

    # Compute majority vote as ground truth
    ground_truth = compute_majority_vote(human_annotations)

    # Determine the unique labels if not provided
    if labels is None:
        all_labels = set(model_coding)
        for annotations in human_annotations.values():
            all_labels.update(annotations)
        # Remove NA values
        all_labels = {
            label for label in all_labels if not pd.isna(label) and label != ""
        }
        # Cast to the expected type for sorted
        labels = sorted(cast(List[Union[int, str]], list(all_labels)))

    # Filter out instances where ground truth is NA
    valid_indices = [
        i for i, gt in enumerate(ground_truth) if not pd.isna(gt) and gt != ""
    ]
    filtered_gt = [ground_truth[i] for i in valid_indices]
    filtered_model = cast(
        Union[List[int], List[str]], [model_coding[i] for i in valid_indices]
    )
    filtered_human: Dict[str, Union[List[int], List[str]]] = {
        rater: cast(
            Union[List[int], List[str]], [annotations[i] for i in valid_indices]
        )
        for rater, annotations in human_annotations.items()
    }

    # Compute class distribution
    class_distribution = Counter(filtered_gt)

    # Initialize results dictionary
    results: ClassificationResults = {
        "class_distribution": dict(class_distribution),
        "global_metrics": {},
        "per_class_metrics": {label: {} for label in labels},
        "ground_truth": ground_truth,
        "confusion_matrices": {},
    }

    # Compute global metrics for the model
    model_accuracy = accuracy_score(filtered_gt, filtered_model)
    model_recall = recall_score(
        filtered_gt, filtered_model, labels=labels, average="macro", zero_division=0
    )
    model_error_rate = 1.0 - model_accuracy

    # Initialize global_metrics if empty
    if "global_metrics" not in results:
        results["global_metrics"] = {}

    # Add model metrics
    model_metrics: GlobalMetrics = {
        "accuracy": model_accuracy,
        "recall": model_recall,
        "error_rate": model_error_rate,
    }
    results["global_metrics"]["model"] = model_metrics

    # Compute per-class metrics for the model
    model_cm = confusion_matrix(filtered_gt, filtered_model, labels=labels)

    # Initialize confusion_matrices if empty
    if "confusion_matrices" not in results:
        results["confusion_matrices"] = {}

    # Add model confusion matrix
    results["confusion_matrices"]["model"] = model_cm

    for i, label in enumerate(labels):
        # True positives are on the diagonal
        true_positives = model_cm[i, i]
        # False negatives are in the row but not on the diagonal
        false_negatives = sum(model_cm[i, :]) - true_positives
        # False positives are in the column but not on the diagonal
        false_positives = sum(model_cm[:, i]) - true_positives

        # Calculate recall and error rate for this class
        class_total = true_positives + false_negatives
        if class_total > 0:
            class_recall = true_positives / class_total
            class_error_rate = false_negatives / class_total
        else:
            class_recall = 0.0
            class_error_rate = 0.0

        # Initialize per_class_metrics for this label if needed
        if label not in results["per_class_metrics"]:
            results["per_class_metrics"][label] = {}

        # Add model metrics for this class
        class_metrics: ClassMetrics = {
            "recall": class_recall,
            "error_rate": class_error_rate,
            "correct_count": int(true_positives),
            "missed_count": int(false_negatives),
            "false_positives": int(false_positives),
        }
        results["per_class_metrics"][label]["model"] = class_metrics

    # Compute metrics for each human annotator
    for rater, annotations in filtered_human.items():
        # Global metrics
        rater_accuracy = accuracy_score(filtered_gt, annotations)
        rater_recall = recall_score(
            filtered_gt, annotations, labels=labels, average="macro", zero_division=0
        )
        rater_error_rate = 1.0 - rater_accuracy

        # Add rater global metrics
        rater_global_metrics: GlobalMetrics = {
            "accuracy": rater_accuracy,
            "recall": rater_recall,
            "error_rate": rater_error_rate,
        }
        results["global_metrics"][rater] = rater_global_metrics

        # Per-class metrics
        rater_cm = confusion_matrix(filtered_gt, annotations, labels=labels)
        results["confusion_matrices"][rater] = rater_cm

        for i, label in enumerate(labels):
            # True positives are on the diagonal
            true_positives = rater_cm[i, i]
            # False negatives are in the row but not on the diagonal
            false_negatives = sum(rater_cm[i, :]) - true_positives
            # False positives are in the column but not on the diagonal
            false_positives = sum(rater_cm[:, i]) - true_positives

            # Calculate recall and error rate for this class
            class_total = true_positives + false_negatives
            if class_total > 0:
                class_recall = true_positives / class_total
                class_error_rate = false_negatives / class_total
            else:
                class_recall = 0.0
                class_error_rate = 0.0

            # Add rater metrics for this class
            rater_class_metrics: ClassMetrics = {
                "recall": class_recall,
                "error_rate": class_error_rate,
                "correct_count": int(true_positives),
                "missed_count": int(false_negatives),
                "false_positives": int(false_positives),
            }
            results["per_class_metrics"][label][rater] = rater_class_metrics

    return results
