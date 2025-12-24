import typing as t


class Field:
    """
    Represents a field in a BaseModel.

    Attributes:
        name: The name of the field.
        type: The type of the field.
        value: The value of the field, or MissingValue if not provided or invalid.
        specs: The field specifications (not implemented yet).
    """

    def __init__(
        self,
        name: str,
        type: type,
        value: t.Any,
        specs: t.Any,
    ) -> None:
        """
        Initialize a Field.

        Args:
            name: The name of the field.
            type: The type of the field.
            value: The value of the field.
            specs: The field specifications.
        """
        self.name = name
        self.type = type
        self.value = value
        self.specs = specs

    def __repr__(self) -> str:
        """Return a string representation of the Field."""
        type_repr = getattr(self.type, "__name__", None)
        if type_repr is None:
            type_repr = repr(self.type)
        return (
            f"Field(name={self.name!r}, type={type_repr}, "
            f"value={self.value!r}, specs={self.specs})"
        )
