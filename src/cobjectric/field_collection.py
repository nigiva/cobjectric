from __future__ import annotations

import typing as t

from cobjectric.field import Field
from cobjectric.path import parse_path
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
        segments = parse_path(path)
        return self._resolve_path(segments)

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

        segment = segments[0]
        remaining = segments[1:]

        # A path cannot start with an index - this is handled at the list level
        if segment.startswith("[") and segment.endswith("]"):
            raise KeyError("Path cannot start with an index. Use field access first.")

        # Access the field
        if segment not in self._fields:
            raise KeyError(f"Field '{segment}' not found in path")

        current = self._fields[segment]

        if not remaining:
            return current

        # Recursive delegation based on type
        return self._resolve_next(current, segment, remaining)

    def _resolve_next(
        self, current: Field | BaseModel, segment: str, remaining: list[str]
    ) -> t.Any:
        """
        Resolve the next segments after accessing a field.

        Args:
            current: The current field or BaseModel instance.
            segment: The segment name that was just accessed.
            remaining: Remaining path segments to resolve.

        Returns:
            The resolved value.

        Raises:
            KeyError: If the path is invalid.
        """
        # Import here to avoid circular import
        from cobjectric.base_model import BaseModel  # noqa: PLC0415

        next_segment = remaining[0]

        # If next segment is an index
        if next_segment.startswith("[") and next_segment.endswith("]"):
            if isinstance(current, Field):
                if not isinstance(current.value, list):
                    raise KeyError(f"Cannot use index on non-list field '{segment}'")
                # Continue with resolution on the list
                return self._resolve_list_path(current.value, remaining)
            elif isinstance(current, list):
                # Case list[list[...]] - continue resolution
                return self._resolve_list_path(current, remaining)
            raise KeyError(f"Cannot use index on non-list field '{segment}'")

        # Otherwise, navigate into nested model via .fields
        if isinstance(current, BaseModel):
            return current.fields._resolve_path(remaining)
        elif isinstance(current, Field):
            if current.value is not MissingValue and isinstance(
                current.value, BaseModel
            ):
                return current.value.fields._resolve_path(remaining)
            raise KeyError(
                f"Cannot access '{next_segment}' on non-model field '{segment}'"
            )

        raise KeyError(f"Cannot access '{next_segment}' on field '{segment}'")

    def _resolve_list_path(self, items: list[t.Any], segments: list[str]) -> t.Any:
        """
        Resolve a path starting from a list.

        Handles list[BaseModel], list[list[...]], list[primitives].

        Args:
            items: The list to resolve from.
            segments: Path segments starting with an index.

        Returns:
            The resolved value.

        Raises:
            KeyError: If the path is invalid.
        """
        # Import here to avoid circular import
        from cobjectric.base_model import BaseModel  # noqa: PLC0415

        segment = segments[0]
        remaining = segments[1:]

        # Extract index
        index_str = segment[1:-1]
        try:
            index = int(index_str)
        except ValueError as e:
            raise KeyError(f"Invalid list index: {index_str}") from e

        try:
            item = items[index]
        except IndexError as e:
            raise KeyError(f"List index {index} out of range") from e

        if not remaining:
            return item

        # Continue resolution based on item type
        next_segment = remaining[0]

        if next_segment.startswith("[") and next_segment.endswith("]"):
            # list[list[...]] - recursion on sub-list
            if isinstance(item, list):
                return self._resolve_list_path(item, remaining)
            raise KeyError(f"Cannot use index on non-list element at index {index}")

        # Access a field of nested model
        if isinstance(item, BaseModel):
            return item.fields._resolve_path(remaining)

        raise KeyError(
            f"Cannot access '{next_segment}' on non-model element at index {index}"
        )

    def __repr__(self) -> str:
        """Return a string representation of the FieldCollection."""
        fields_repr = ", ".join(f"{name}=..." for name in self._fields.keys())
        return f"FieldCollection({fields_repr})"
