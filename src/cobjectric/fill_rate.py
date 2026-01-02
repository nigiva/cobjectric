from __future__ import annotations

import typing as t
from dataclasses import dataclass

from cobjectric.exceptions import InvalidWeightError
from cobjectric.sentinel import MissingValue

FillRateFunc = t.Callable[[t.Any], float]


def not_missing_fill_rate(value: t.Any) -> float:
    """
    Fill rate function: returns 0.0 if MissingValue, else 1.0.

    Args:
        value: The field value.

    Returns:
        float: 0.0 if MissingValue, 1.0 otherwise.

    Examples:
        >>> from cobjectric.fill_rate import not_missing_fill_rate
        >>> from cobjectric import MissingValue
        >>> not_missing_fill_rate("John")
        1.0
        >>> not_missing_fill_rate(MissingValue)
        0.0
    """
    return 0.0 if value is MissingValue else 1.0


default_fill_rate_func = not_missing_fill_rate


@dataclass
class FillRateFuncInfo:
    """
    Stores fill_rate_func info attached to a method.

    Attributes:
        field_patterns: Tuple of field names or glob patterns to match.
        func: The fill_rate_func to apply.
        weight: Weight for fill rate computation (default: 1.0, must be >= 0.0).
    """

    field_patterns: tuple[str, ...]
    func: FillRateFunc
    weight: float = 1.0


def fill_rate_func(
    *field_patterns: str,
    weight: float = 1.0,
) -> t.Callable[[FillRateFunc], FillRateFunc]:
    """
    Decorator to define a fill_rate_func for one or more fields.

    Args:
        *field_patterns: Field names or glob patterns (e.g., "name", "email", "name_*")
        weight: Weight for fill rate computation (default: 1.0, must be >= 0.0).

    Returns:
        Decorated function

    Raises:
        InvalidWeightError: If weight is negative (< 0.0).

    Example:
        ```python
        class Person(BaseModel):
            name: str
            email: str

            @fill_rate_func("name", "email", weight=2.0)
            def fill_rate_name_email(x: t.Any) -> float:
                return len(x) / 100 if x is not MissingValue else 0.0
        ```
    """
    if weight < 0.0:
        raise InvalidWeightError(weight, "decorator")

    def decorator(func: FillRateFunc) -> FillRateFunc:
        if not hasattr(func, "_fill_rate_funcs"):
            func._fill_rate_funcs = []  # type: ignore[attr-defined]
        func._fill_rate_funcs.append(  # type: ignore[attr-defined]
            FillRateFuncInfo(field_patterns, func, weight)
        )
        return func

    return decorator


FillRateAccuracyFunc = t.Callable[[t.Any, t.Any], float]


def same_state_fill_rate_accuracy(got: t.Any, expected: t.Any) -> float:
    """
    Fill rate accuracy function: returns 1.0 if both have same state, else 0.0.

    Args:
        got: The field value from the model being evaluated.
        expected: The field value from the expected model.

    Returns:
        float: 1.0 if both are filled or both are MissingValue, 0.0 otherwise.

    Examples:
        >>> from cobjectric.fill_rate import same_state_fill_rate_accuracy
        >>> from cobjectric import MissingValue
        >>> same_state_fill_rate_accuracy("John", "Jane")
        1.0
        >>> same_state_fill_rate_accuracy("John", MissingValue)
        0.0
        >>> same_state_fill_rate_accuracy(MissingValue, MissingValue)
        1.0
    """
    got_filled = got is not MissingValue
    expected_filled = expected is not MissingValue
    return 1.0 if got_filled == expected_filled else 0.0


default_fill_rate_accuracy_func = same_state_fill_rate_accuracy


@dataclass
class FillRateAccuracyFuncInfo:
    """
    Stores fill_rate_accuracy_func info attached to a method.

    Attributes:
        field_patterns: Tuple of field names or glob patterns to match.
        func: The fill_rate_accuracy_func to apply.
        weight: Weight for fill rate accuracy computation
            (default: 1.0, must be >= 0.0).
    """

    field_patterns: tuple[str, ...]
    func: FillRateAccuracyFunc
    weight: float = 1.0


def fill_rate_accuracy_func(
    *field_patterns: str,
    weight: float = 1.0,
) -> t.Callable[[FillRateAccuracyFunc], FillRateAccuracyFunc]:
    """
    Decorator to define a fill_rate_accuracy_func for one or more fields.

    Args:
        *field_patterns: Field names or glob patterns (e.g., "name", "email", "name_*")
        weight: Weight for fill rate accuracy computation
            (default: 1.0, must be >= 0.0).

    Returns:
        Decorated function

    Raises:
        InvalidWeightError: If weight is negative (< 0.0).

    Example:
        ```python
        class Person(BaseModel):
            name: str
            email: str

            @fill_rate_accuracy_func("name", "email", weight=2.0)
            def accuracy_name_email(got: t.Any, expected: t.Any) -> float:
                return (
                    1.0
                    if (got is not MissingValue) == (expected is not MissingValue)
                    else 0.0
                )
        ```
    """
    if weight < 0.0:
        raise InvalidWeightError(weight, "decorator", "fill_rate_accuracy")

    def decorator(func: FillRateAccuracyFunc) -> FillRateAccuracyFunc:
        if not hasattr(func, "_fill_rate_accuracy_funcs"):
            func._fill_rate_accuracy_funcs = []  # type: ignore[attr-defined]
        func._fill_rate_accuracy_funcs.append(  # type: ignore[attr-defined]
            FillRateAccuracyFuncInfo(field_patterns, func, weight)
        )
        return func

    return decorator


SimilarityFunc = t.Callable[[t.Any, t.Any], float]


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
