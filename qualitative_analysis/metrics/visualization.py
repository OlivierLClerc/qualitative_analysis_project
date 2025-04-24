"""
visualization.py

This module provides functions for visualizing metrics.

Functions:
    - plot_confusion_matrices(model_coding, human_annotations, labels):
      Generates confusion matrices comparing model predictions to human annotations
      and between human annotators, visualized as heatmaps.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Union


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
