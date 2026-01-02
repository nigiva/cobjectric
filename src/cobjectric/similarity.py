from __future__ import annotations

import typing as t
from dataclasses import dataclass
from datetime import datetime, timedelta

from rapidfuzz import fuzz

from cobjectric.exceptions import InvalidWeightError

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


def datetime_similarity_factory(
    max_difference: timedelta | None = None,
) -> SimilarityFunc:
    """
    Factory for datetime similarity based on time difference.

    Args:
        max_difference: Maximum time difference as timedelta for comparison.
            - None (default): Exact match only (1.0 if a == b, 0.0 otherwise)
            - timedelta > 0: Gradual similarity based on time difference
              Formula: max(0.0, 1.0 - |time_diff| / max_difference)

    Returns:
        A similarity function that compares two datetime string values (ISO format).

    Examples:
        >>> from datetime import timedelta
        >>> # Exact match required
        >>> exact = datetime_similarity_factory()
        >>> exact("2024-01-15T10:30:00Z", "2024-01-15T10:30:00Z")
        1.0
        >>> exact("2024-01-15T10:30:00Z", "2024-01-15T10:30:01Z")
        0.0
        >>>
        >>> # Gradual decrease: up to 1 hour of difference
        >>> gradual = datetime_similarity_factory(max_difference=timedelta(hours=1))
        >>> gradual("2024-01-15T10:00:00Z", "2024-01-15T10:00:00Z")
        1.0
        >>> gradual("2024-01-15T10:00:00Z", "2024-01-15T10:30:00Z")  # 30 min
        ... # diff=30min, 30min/60min=0.5, 1-0.5=0.5
        0.5
        >>> gradual("2024-01-15T10:00:00Z", "2024-01-15T11:00:00Z")  # 1 hour
        ... # diff=1h, 1h/1h=1.0, 1-1.0=0.0
        0.0

    Raises:
        ValueError: If max_difference is <= 0.
    """
    if max_difference is not None and max_difference.total_seconds() <= 0:
        raise ValueError("max_difference must be > 0")

    def parse_datetime(value: t.Any) -> datetime | None:
        """Parse datetime from string, trying common ISO formats."""
        assert value is not None  # Already handled by datetime_similarity
        if isinstance(value, datetime):
            return value

        if not isinstance(value, str):
            return None

        # Try common ISO formats
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

        return None

    def datetime_similarity(a: t.Any, b: t.Any) -> float:
        if a is None or b is None:
            return 0.0

        dt_a = parse_datetime(a)
        dt_b = parse_datetime(b)

        if dt_a is None or dt_b is None:
            return 0.0

        if max_difference is None:
            return 1.0 if dt_a == dt_b else 0.0

        diff = abs(dt_a - dt_b)
        max_diff_seconds = max_difference.total_seconds()
        diff_seconds = diff.total_seconds()
        return max(0.0, 1.0 - diff_seconds / max_diff_seconds)

    return datetime_similarity


@dataclass
class SimilarityFuncInfo:
    """
    Stores similarity_func info attached to a method.

    Attributes:
        field_patterns: Tuple of field names or glob patterns to match.
        func: The similarity_func to apply.
        weight: Weight for similarity computation (default: 1.0, must be >= 0.0).
    """

    field_patterns: tuple[str, ...]
    func: SimilarityFunc
    weight: float = 1.0


def similarity_func(
    *field_patterns: str,
    weight: float = 1.0,
) -> t.Callable[[SimilarityFunc], SimilarityFunc]:
    """
    Decorator to define a similarity_func for one or more fields.

    Args:
        *field_patterns: Field names or glob patterns (e.g., "name", "email", "name_*")
        weight: Weight for similarity computation (default: 1.0, must be >= 0.0).

    Returns:
        Decorated function

    Raises:
        InvalidWeightError: If weight is negative (< 0.0).

    Example:
        ```python
        class Person(BaseModel):
            name: str
            email: str

            @similarity_func("name", "email", weight=2.0)
            def similarity_name_email(x: t.Any, y: t.Any) -> float:
                return 1.0 if x == y else 0.0
        ```
    """
    if weight < 0.0:
        raise InvalidWeightError(weight, "decorator", "similarity")

    def decorator(func: SimilarityFunc) -> SimilarityFunc:
        if not hasattr(func, "_similarity_funcs"):
            func._similarity_funcs = []  # type: ignore[attr-defined]
        func._similarity_funcs.append(  # type: ignore[attr-defined]
            SimilarityFuncInfo(field_patterns, func, weight)
        )
        return func

    return decorator
