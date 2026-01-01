import inspect
import typing as t
from dataclasses import dataclass, field

from cobjectric.exceptions import InvalidWeightError
from cobjectric.fill_rate import (
    FillRateAccuracyFunc,
    FillRateFunc,
    SimilarityFunc,
    not_missing_fill_rate,
    same_state_fill_rate_accuracy,
)
from cobjectric.list_compare import ListCompareStrategy
from cobjectric.similarities import exact_similarity

if t.TYPE_CHECKING:
    from cobjectric.context import FieldContext

Normalizer = t.Callable[[t.Any], t.Any]
ContextualNormalizer = t.Callable[[t.Any, "FieldContext"], t.Any]
AnyNormalizer = Normalizer | ContextualNormalizer


def is_contextual_normalizer(func: AnyNormalizer) -> bool:
    """
    Check if normalizer expects context (2 params) or not (1 param).

    Args:
        func: The normalizer function to check.

    Returns:
        True if the function expects 2 parameters (value, context),
        False if it expects 1 parameter (value only).
    """
    sig = inspect.signature(func)
    return len(sig.parameters) >= 2


@dataclass
class FieldSpec:
    """
    Specification for a field in a BaseModel.
    """

    metadata: dict[str, t.Any] = field(default_factory=dict)
    normalizer: AnyNormalizer | None = None
    fill_rate_func: FillRateFunc = field(default=not_missing_fill_rate)
    fill_rate_weight: float = 1.0
    fill_rate_accuracy_func: FillRateAccuracyFunc = field(
        default=same_state_fill_rate_accuracy
    )
    fill_rate_accuracy_weight: float = 1.0
    similarity_func: SimilarityFunc = field(default=exact_similarity)
    similarity_weight: float = 1.0
    list_compare_strategy: ListCompareStrategy = field(
        default=ListCompareStrategy.PAIRWISE
    )


# We use a function instead of directly using FieldSpec because:
# - Writing `name: str = FieldSpec()` would cause a type error (FieldSpec is not str)
# - By returning `Any`, mypy accepts `name: str = Spec()` without complaint
def Spec(  # noqa: N802
    metadata: dict[str, t.Any] | None = None,
    normalizer: AnyNormalizer | None = None,
    fill_rate_func: FillRateFunc | None = None,
    fill_rate_weight: float = 1.0,
    fill_rate_accuracy_func: FillRateAccuracyFunc | None = None,
    fill_rate_accuracy_weight: float = 1.0,
    similarity_func: SimilarityFunc | None = None,
    similarity_weight: float = 1.0,
    list_compare_strategy: ListCompareStrategy | str | None = None,
) -> t.Any:
    """
    Create a FieldSpec for a field.

    Args:
        metadata (dict[str, Any] | None): Optional metadata for the field.
        normalizer (AnyNormalizer | None): Optional normalizer function for the field.
            Can be a simple normalizer (1 param: value) or contextual
            (2 params: value, context).
        fill_rate_func (FillRateFunc | None): Optional fill rate function for the field.
        fill_rate_weight (float): Weight for fill rate computation
            (default: 1.0, must be >= 0.0).
        fill_rate_accuracy_func (FillRateAccuracyFunc | None): Optional
            fill rate accuracy function.
        fill_rate_accuracy_weight (float): Weight for fill rate accuracy computation
            (default: 1.0, must be >= 0.0).
        similarity_func (SimilarityFunc | None): Optional similarity function.
        similarity_weight (float): Weight for similarity computation
            (default: 1.0, must be >= 0.0).
        list_compare_strategy (ListCompareStrategy | str | None): Strategy for
            comparing list[BaseModel] items. Valid values: "pairwise", "levenshtein",
            "optimal_assignment", or ListCompareStrategy enum values.
            Only valid for list[BaseModel] fields (default: PAIRWISE).

    Returns:
        Any: A FieldSpec instance (typed as Any for type checker compatibility).

    Raises:
        InvalidWeightError: If weight is negative (< 0.0).
        ValueError: If list_compare_strategy is an invalid string value.
    """
    if fill_rate_weight < 0.0:
        raise InvalidWeightError(fill_rate_weight, "Spec", "fill_rate")
    if fill_rate_accuracy_weight < 0.0:
        raise InvalidWeightError(
            fill_rate_accuracy_weight, "Spec", "fill_rate_accuracy"
        )
    if similarity_weight < 0.0:
        raise InvalidWeightError(similarity_weight, "Spec", "similarity")

    # Convert string to enum if needed
    strategy: ListCompareStrategy = ListCompareStrategy.PAIRWISE
    if list_compare_strategy is not None:
        if isinstance(list_compare_strategy, ListCompareStrategy):
            strategy = list_compare_strategy
        elif isinstance(list_compare_strategy, str):
            try:
                strategy = ListCompareStrategy(list_compare_strategy)
            except ValueError as e:
                raise ValueError(
                    f"Invalid list_compare_strategy: {list_compare_strategy}. "
                    f"Valid values: {[s.value for s in ListCompareStrategy]}"
                ) from e
        else:
            raise ValueError(
                f"list_compare_strategy must be str or ListCompareStrategy, "
                f"got {type(list_compare_strategy).__name__}"
            )

    return FieldSpec(
        metadata=metadata if metadata is not None else {},
        normalizer=normalizer,
        fill_rate_func=(
            fill_rate_func if fill_rate_func is not None else not_missing_fill_rate
        ),
        fill_rate_weight=fill_rate_weight,
        fill_rate_accuracy_func=(
            fill_rate_accuracy_func
            if fill_rate_accuracy_func is not None
            else same_state_fill_rate_accuracy
        ),
        fill_rate_accuracy_weight=fill_rate_accuracy_weight,
        similarity_func=(
            similarity_func if similarity_func is not None else exact_similarity
        ),
        similarity_weight=similarity_weight,
        list_compare_strategy=strategy,
    )
