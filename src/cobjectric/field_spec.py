import typing as t
from dataclasses import dataclass, field


@dataclass
class FieldSpec:
    """
    Specification for a field in a BaseModel.
    """

    metadata: dict[str, t.Any] = field(default_factory=dict)


# We use a function instead of directly using FieldSpec because:
# - Writing `name: str = FieldSpec()` would cause a type error (FieldSpec is not str)
# - By returning `Any`, mypy accepts `name: str = Spec()` without complaint
def Spec(metadata: dict[str, t.Any] | None = None) -> t.Any:  # noqa: N802
    """
    Create a FieldSpec for a field.

    Args:
        metadata (dict[str, Any] | None): Optional metadata for the field.

    Returns:
        Any: A FieldSpec instance (typed as Any for type checker compatibility).
    """
    return FieldSpec(metadata=metadata if metadata is not None else {})
