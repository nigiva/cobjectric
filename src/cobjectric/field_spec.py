import typing as t
from dataclasses import dataclass, field

Normalizer = t.Callable[[t.Any], t.Any]


@dataclass
class FieldSpec:
    """
    Specification for a field in a BaseModel.
    """

    metadata: dict[str, t.Any] = field(default_factory=dict)
    normalizer: Normalizer | None = None


# We use a function instead of directly using FieldSpec because:
# - Writing `name: str = FieldSpec()` would cause a type error (FieldSpec is not str)
# - By returning `Any`, mypy accepts `name: str = Spec()` without complaint
def Spec(  # noqa: N802
    metadata: dict[str, t.Any] | None = None,
    normalizer: Normalizer | None = None,
) -> t.Any:
    """
    Create a FieldSpec for a field.

    Args:
        metadata (dict[str, Any] | None): Optional metadata for the field.
        normalizer (Normalizer | None): Optional normalizer function for the field.

    Returns:
        Any: A FieldSpec instance (typed as Any for type checker compatibility).
    """
    return FieldSpec(
        metadata=metadata if metadata is not None else {},
        normalizer=normalizer,
    )
