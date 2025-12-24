import typing as t

from cobjectric.field import Field


class FieldCollection:
    """
    Collection of fields for a BaseModel instance.

    Provides attribute-based access to fields.
    """

    def __init__(self, fields: dict[str, Field]) -> None:
        """
        Initialize a FieldCollection.

        Args:
            fields: Dictionary mapping field names to Field instances.
        """
        self._fields = fields

    def __getattr__(self, name: str) -> Field:
        """
        Get a field by name.

        Args:
            name: The name of the field.

        Returns:
            The Field instance.

        Raises:
            AttributeError: If the field does not exist.
        """
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __iter__(self) -> t.Iterator[Field]:
        """Iterate over all fields."""
        return iter(self._fields.values())

    def __repr__(self) -> str:
        """Return a string representation of the FieldCollection."""
        fields_repr = ", ".join(
            f"{name}={field!r}" for name, field in self._fields.items()
        )
        return f"FieldCollection({fields_repr})"
