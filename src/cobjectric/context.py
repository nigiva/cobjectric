import typing as t
from dataclasses import dataclass

if t.TYPE_CHECKING:
    from cobjectric.field_spec import FieldSpec


@dataclass
class FieldContext:
    """
    Context information passed to contextual normalizers.

    Attributes:
        name: The field name.
        field_type: The declared Python type of the field.
        spec: The FieldSpec associated with the field.
    """

    name: str
    field_type: type
    spec: "FieldSpec"
