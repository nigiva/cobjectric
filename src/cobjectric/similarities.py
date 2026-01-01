import typing as t

from rapidfuzz import fuzz

SimilarityFunc = t.Callable[[t.Any, t.Any], float]


def exact_similarity(a: t.Any, b: t.Any) -> float:
    """
    Exact equality similarity.

    Returns 1.0 if a == b, else 0.0.
    Works with any type (str, int, float, bool, etc.)

    Args:
        a: First value to compare.
        b: Second value to compare.

    Returns:
        float: 1.0 if values are equal, 0.0 otherwise.

    Examples:
        >>> exact_similarity("John", "John")
        1.0
        >>> exact_similarity("John", "Jane")
        0.0
        >>> exact_similarity(10, 10)
        1.0
        >>> exact_similarity(10, 11)
        0.0
    """
    return 1.0 if a == b else 0.0


def fuzzy_similarity_factory(
    scorer: str = "ratio",
) -> SimilarityFunc:
    """
    Factory to create fuzzy string similarity using rapidfuzz.

    Args:
        scorer: The rapidfuzz.fuzz scorer to use. Options:
            - "ratio" (default): Standard Levenshtein ratio
            - "partial_ratio": Best partial match ratio
            - "token_sort_ratio": Token sorted ratio
            - "token_set_ratio": Token set ratio
            - "WRatio": Weighted ratio (smart combination)
            - "QRatio": Quick ratio

    Returns:
        A similarity function that compares two string values.

    Examples:
        >>> fuzzy = fuzzy_similarity_factory()
        >>> fuzzy("John Doe", "John Doe")
        1.0
        >>> fuzzy("John Doe", "john doe")  # case difference
        0.9...
        >>> fuzzy("John Doe", "Jane Doe")
        0.75...
    """
    scorer_func = getattr(fuzz, scorer)

    def fuzzy_similarity(a: t.Any, b: t.Any) -> float:
        if a is None or b is None:
            return 0.0
        return scorer_func(str(a), str(b)) / 100.0

    return fuzzy_similarity


def numeric_similarity_factory(
    max_difference: float | None = None,
) -> SimilarityFunc:
    """
    Factory for numeric similarity based on difference.

    Args:
        max_difference: Maximum absolute difference for comparison.
            - None (default): Exact match only (1.0 if a == b, 0.0 otherwise)
            - float > 0: Gradual similarity based on difference
              Formula: max(0.0, 1.0 - |a - b| / max_difference)

    Returns:
        A similarity function that compares two numeric values.

    Examples:
        >>> # Exact match required
        >>> exact = numeric_similarity_factory()
        >>> exact(10, 10)
        1.0
        >>> exact(10, 11)
        0.0
        >>>
        >>> # Gradual decrease: up to 5 units of difference
        >>> gradual = numeric_similarity_factory(max_difference=5.0)
        >>> gradual(10, 10)
        1.0
        >>> gradual(10, 12)  # diff=2, 2/5=0.4, 1-0.4=0.6
        0.6
        >>> gradual(10, 15)  # diff=5, 5/5=1.0, 1-1.0=0.0
        0.0
        >>> gradual(10, 20)  # diff>max, capped at 0
        0.0

    Raises:
        ValueError: If max_difference is <= 0.
    """
    if max_difference is not None and max_difference <= 0:
        raise ValueError("max_difference must be > 0")

    def numeric_similarity(a: t.Any, b: t.Any) -> float:
        if a is None or b is None:
            return 0.0

        try:
            a_num = float(a)
            b_num = float(b)
        except (ValueError, TypeError):
            return 0.0

        if max_difference is None:
            return 1.0 if a_num == b_num else 0.0

        diff = abs(a_num - b_num)
        return max(0.0, 1.0 - diff / max_difference)

    return numeric_similarity
