import typing as t
from dataclasses import dataclass


@dataclass
class FieldNormalizerInfo:
    """
    Stores normalizer info attached to a method.

    Attributes:
        field_patterns: Tuple of field names or glob patterns to match.
        func: The normalizer function to apply.
    """

    field_patterns: tuple[str, ...]
    func: t.Callable[..., t.Any]


def field_normalizer(
    *field_patterns: str,
) -> t.Callable[[t.Callable[..., t.Any]], t.Callable[..., t.Any]]:
    """
    Decorator to define a normalizer for one or more fields.

    Args:
        *field_patterns: Field names or glob patterns (e.g., "name", "email", "name_*")

    Returns:
        Decorated function
    """

    def decorator(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
        if not hasattr(func, "_field_normalizers"):
            func._field_normalizers = []  # type: ignore[attr-defined]
        func._field_normalizers.append(  # type: ignore[attr-defined]
            FieldNormalizerInfo(field_patterns, func)
        )
        return func

    return decorator
