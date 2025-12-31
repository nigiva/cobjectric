import typing as t
from dataclasses import dataclass, field

from cobjectric.exceptions import InvalidWeightError
from cobjectric.fill_rate import FillRateAccuracyFunc, FillRateFunc

Normalizer = t.Callable[[t.Any], t.Any]


@dataclass
class FieldSpec:
    """
    Specification for a field in a BaseModel.
    """

    metadata: dict[str, t.Any] = field(default_factory=dict)
    normalizer: Normalizer | None = None
    fill_rate_func: FillRateFunc | None = None
    fill_rate_weight: float = 1.0
    fill_rate_accuracy_func: FillRateAccuracyFunc | None = None
    fill_rate_accuracy_weight: float = 1.0


# We use a function instead of directly using FieldSpec because:
# - Writing `name: str = FieldSpec()` would cause a type error (FieldSpec is not str)
# - By returning `Any`, mypy accepts `name: str = Spec()` without complaint
def Spec(  # noqa: N802
    metadata: dict[str, t.Any] | None = None,
    normalizer: Normalizer | None = None,
    fill_rate_func: FillRateFunc | None = None,
    fill_rate_weight: float = 1.0,
    fill_rate_accuracy_func: FillRateAccuracyFunc | None = None,
    fill_rate_accuracy_weight: float = 1.0,
) -> t.Any:
    """
    Create a FieldSpec for a field.

    Args:
        metadata (dict[str, Any] | None): Optional metadata for the field.
        normalizer (Normalizer | None): Optional normalizer function for the field.
        fill_rate_func (FillRateFunc | None): Optional fill rate function for the field.
        fill_rate_weight (float): Weight for fill rate computation
            (default: 1.0, must be >= 0.0).
        fill_rate_accuracy_func (FillRateAccuracyFunc | None): Optional
            fill rate accuracy function.
        fill_rate_accuracy_weight (float): Weight for fill rate accuracy computation
            (default: 1.0, must be >= 0.0).

    Returns:
        Any: A FieldSpec instance (typed as Any for type checker compatibility).

    Raises:
        InvalidWeightError: If weight is negative (< 0.0).
    """
    if fill_rate_weight < 0.0:
        raise InvalidWeightError(fill_rate_weight, "Spec", "fill_rate")
    if fill_rate_accuracy_weight < 0.0:
        raise InvalidWeightError(
            fill_rate_accuracy_weight, "Spec", "fill_rate_accuracy"
        )
    return FieldSpec(
        metadata=metadata if metadata is not None else {},
        normalizer=normalizer,
        fill_rate_func=fill_rate_func,
        fill_rate_weight=fill_rate_weight,
        fill_rate_accuracy_func=fill_rate_accuracy_func,
        fill_rate_accuracy_weight=fill_rate_accuracy_weight,
    )
