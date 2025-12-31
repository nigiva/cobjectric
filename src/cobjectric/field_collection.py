from __future__ import annotations

import typing as t

from cobjectric.field import Field
from cobjectric.sentinel import MissingValue

if t.TYPE_CHECKING:  # pragma: no cover
    from cobjectric.base_model import BaseModel


class FieldCollection:
    """
    Collection of fields for a BaseModel instance.

    Provides attribute-based access to fields.
    """

    def __init__(self, fields: dict[str, Field | BaseModel]) -> None:
        """
        Initialize a FieldCollection.

        Args:
            fields: Dictionary mapping field names to Field or BaseModel
                instances.
        """
        self._fields = fields

    def __getattr__(self, name: str) -> Field | BaseModel:
        """
        Get a field by name.

        Args:
            name: The name of the field.

        Returns:
            The Field instance or BaseModel instance.

        Raises:
            AttributeError: If the field does not exist.
        """
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __iter__(self) -> t.Iterator[Field | BaseModel]:
        """Iterate over all fields."""
        return iter(self._fields.values())

    def __getitem__(self, path: str) -> Field | BaseModel | t.Any:
        """
        Get a field by path.

        Args:
            path: Path to the field (e.g., "name", "address.city", "items[0].name").

        Returns:
            The Field instance, BaseModel instance, or value.

        Raises:
            KeyError: If the path is invalid.
        """
        segments = self._parse_path(path)
        return self._resolve_path(segments)

    def _parse_path(self, path: str) -> list[str]:
        """
        Parse a path string into segments.

        Args:
            path: Path string (e.g., "address.city", "items[0].name").

        Returns:
            List of path segments.
        """
        segments: list[str] = []
        current = ""
        i = 0
        while i < len(path):
            if path[i] == ".":
                if current:
                    segments.append(current)
                    current = ""
            elif path[i] == "[":
                if current:
                    segments.append(current)
                    current = ""
                # Find closing bracket
                j = i + 1
                while j < len(path) and path[j] != "]":
                    j += 1
                if j >= len(path):
                    raise KeyError(f"Invalid path: {path}")
                index_str = path[i + 1 : j]
                try:
                    index = int(index_str)
                    segments.append(f"[{index}]")
                except ValueError as e:
                    raise KeyError(f"Invalid path: {path}") from e
                i = j
            else:
                current += path[i]
            i += 1
        if current:
            segments.append(current)
        return segments

    def _resolve_path(self, segments: list[str]) -> Field | BaseModel | t.Any:
        """
        Resolve a path from segments.

        Args:
            segments: List of path segments.

        Returns:
            The Field instance, BaseModel instance, or value.

        Raises:
            KeyError: If the path is invalid.
        """
        if not segments:
            raise KeyError("Empty path")

        current: Field | BaseModel | t.Any | None = None
        current_fields = self._fields

        for i, segment in enumerate(segments):
            if segment.startswith("[") and segment.endswith("]"):
                # List index - not yet supported
                raise KeyError(f"List index access not yet supported: {segment}")

            if segment not in current_fields:
                raise KeyError(f"Field '{segment}' not found in path")

            current = current_fields[segment]

            if i < len(segments) - 1:
                # More segments to go
                next_segment = segments[i + 1]
                # Check if next segment is a list index
                if next_segment.startswith("[") and next_segment.endswith("]"):
                    # List index - not yet supported
                    raise KeyError(
                        f"List index access not yet supported: {next_segment}"
                    )

                # Import here to avoid circular import
                from cobjectric.base_model import BaseModel  # noqa: PLC0415

                if isinstance(current, BaseModel):
                    current_fields = current._fields
                elif isinstance(current, Field):
                    # Check if it's a nested model type
                    if current.value is not MissingValue and isinstance(
                        current.value, BaseModel
                    ):
                        current_fields = current.value._fields
                    else:
                        raise KeyError(
                            f"Cannot access '{next_segment}' "
                            f"on non-model field '{segment}'"
                        )
                else:
                    raise KeyError(
                        f"Cannot access '{next_segment}' on field '{segment}'"
                    )

        assert current is not None, "Invalid path"

        # Return the object itself (Field or BaseModel)
        # The test expects to access .value on Field, so return Field, not Field.value
        return current

    def __repr__(self) -> str:
        """Return a string representation of the FieldCollection."""
        fields_repr = ", ".join(
            f"{name}={field!r}" for name, field in self._fields.items()
        )
        return f"FieldCollection({fields_repr})"
