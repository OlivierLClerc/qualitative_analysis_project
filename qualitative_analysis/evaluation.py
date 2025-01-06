# evaluation.py
from sklearn.metrics import cohen_kappa_score

def compute_cohens_kappa(judgments_1, judgments_2, labels=None, weights=None):
    """
    Computes Cohen's Kappa score between two sets of judgments.

    Cohen's Kappa is a statistic that measures inter-rater agreement for qualitative (categorical) items.
    It is generally thought to be a more robust measure than simple percent agreement calculation,
    since it takes into account the agreement occurring by chance.

    Parameters:
        judgments_1 (list or array-like): Labels from the first rater.
        judgments_2 (list or array-like): Labels from the second rater.
        labels (list, optional): List of labels to index the confusion matrix.
                                 By default, uses the union of labels in `judgments_1` and `judgments_2`.
        weights (str, optional): Weighting type to calculate the score.
                                 Options are 'linear', 'quadratic', or None (default).

    Returns:
        float: Cohen's Kappa score ranging from -1 (complete disagreement) to 1 (complete agreement).

    Raises:
        ValueError: If input lists have different lengths or contain invalid labels.

    Example:
        Basic usage without specifying labels or weights:

        >>> from qualitative_analysis.evaluation import compute_cohens_kappa
        >>> judgments_1 = [0, 1, 2, 1, 0]
        >>> judgments_2 = [0, 2, 2, 1, 0]
        >>> kappa = compute_cohens_kappa(judgments_1, judgments_2)
        >>> print(f"Cohen's Kappa Score: {kappa:.2f}")
        Cohen's Kappa Score: 0.71

        Using specified labels and linear weights:

        >>> labels = [0, 1, 2]
        >>> kappa_linear = compute_cohens_kappa(judgments_1, judgments_2, labels=labels, weights='linear')
        >>> print(f"Cohen's Kappa Score with linear weights: {kappa_linear:.2f}")
        Cohen's Kappa Score with linear weights: 0.78

        Handling a case with perfect agreement:

        >>> judgments_3 = [0, 1, 2, 1, 0]
        >>> kappa_perfect = compute_cohens_kappa(judgments_1, judgments_3)
        >>> print(f"Cohen's Kappa Score with perfect agreement: {kappa_perfect:.2f}")
        Cohen's Kappa Score with perfect agreement: 1.00

        Example with no agreement beyond chance:

        >>> judgments_4 = [2, 0, 1, 2, 1]
        >>> kappa_none = compute_cohens_kappa(judgments_1, judgments_4)
        >>> print(f"Cohen's Kappa Score with no agreement: {kappa_none:.2f}")
        Cohen's Kappa Score with no agreement: -0.40

    References:
        - Cohen, J. (1960). A coefficient of agreement for nominal scales.
          Educational and Psychological Measurement, 20(1), 37-46.
    """
    return cohen_kappa_score(judgments_1, judgments_2, labels=labels, weights=weights)

def compute_all_kappas(model_coding, human_annotations, labels=None, weights=None, verbose=False):
    """
    Computes Cohen's Kappa score for all combinations of model coding
    and human annotations, as well as between human annotations themselves.

    Parameters:
        model_coding (list): List of model predictions.
        human_annotations (dict): Dictionary where keys are rater names and values are lists of annotations.
        labels (list, optional): List of labels to index the confusion matrix.
        weights (str, optional): Weighting type to calculate the score.
        verbose (bool, optional): If True, prints the Kappa scores. Default is False.

    Returns:
        dict: Dictionary containing Cohen's Kappa scores for all combinations.
    """
    results = {}

    # Compare model with each human rater
    for rater, annotations in human_annotations.items():
        kappa = compute_cohens_kappa(model_coding, annotations, labels=labels, weights=weights)
        results[f'model_vs_{rater}'] = kappa
        if verbose:
            print(f"Model vs {rater}: {kappa:.2f}")

    # Compare each human rater with every other human rater
    raters = list(human_annotations.keys())
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            rater1, rater2 = raters[i], raters[j]
            kappa = compute_cohens_kappa(human_annotations[rater1], human_annotations[rater2], labels=labels, weights=weights)
            results[f'{rater1}_vs_{rater2}'] = kappa
            if verbose:
                print(f"{rater1} vs {rater2}: {kappa:.2f}")

    return results
