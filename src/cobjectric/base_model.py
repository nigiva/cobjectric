from __future__ import annotations

import types
import typing as t

from cobjectric.exceptions import (
    MissingListTypeArgError,
    UnsupportedListTypeError,
    UnsupportedTypeError,
)
from cobjectric.field import Field
from cobjectric.field_collection import FieldCollection
from cobjectric.sentinel import MissingValue


class BaseModel:
    """
    Base class for models with typed fields.

    Fields are defined as class attributes with type annotations.
    Fields can be accessed via the .fields attribute, which provides
    readonly access to Field instances.
    """

    _fields: dict[str, Field | BaseModel]
    _initialized: bool

    @staticmethod
    def _validate_field_type(field_type: type) -> None:
        """
        Validate that a field type is supported.

        Args:
            field_type: The type to validate.

        Raises:
            UnsupportedTypeError: If the type is not JSON-compatible.
            MissingListTypeArgError: If list is used without type arguments.
        """
        if field_type is list or field_type is t.List:
            args = t.get_args(field_type)
            if not args:
                raise MissingListTypeArgError()

        origin = t.get_origin(field_type)

        if origin is list:
            args = t.get_args(field_type)
            if not args:
                raise MissingListTypeArgError()

        if field_type is t.Any or field_type is object:
            raise UnsupportedTypeError(field_type)

        if origin is set or origin is tuple or origin is frozenset:
            raise UnsupportedTypeError(field_type)

    @staticmethod
    def _is_list_type(field_type: type) -> bool:
        """
        Check if a field type is a list type.

        Args:
            field_type: The type to check.

        Returns:
            True if the type is a list, False otherwise.
        """
        if field_type is list or field_type is t.List:
            return True
        origin = t.get_origin(field_type)
        return origin is list

    @staticmethod
    def _get_list_element_type(field_type: type) -> type:
        """
        Extract the element type from a list type.

        Args:
            field_type: The list type.

        Returns:
            The element type.

        Raises:
            MissingListTypeArgError: If list has no type arguments.
        """
        args = t.get_args(field_type)
        if not args:
            raise MissingListTypeArgError()
        return args[0]

    @classmethod
    def _process_list_value(
        cls, value: t.Any, element_type: type
    ) -> list[t.Any] | t.Any:
        """
        Process a list value with partial filtering.

        Args:
            value: The value to process (should be a list).
            element_type: The expected element type.

        Returns:
            A filtered list with only valid elements, or MissingValue if the list
            is empty after filtering or if value is not a list.
        """
        if not isinstance(value, list):
            return MissingValue

        if not value:
            return []

        is_nested_model = (
            element_type is not t.Any
            and isinstance(element_type, type)
            and issubclass(element_type, BaseModel)
        )

        validated_list = []
        for item in value:
            if is_nested_model and element_type is not t.Any:
                if isinstance(item, dict):
                    try:
                        validated_list.append(
                            element_type.from_dict(item)  # type: ignore[attr-defined]
                        )
                    except Exception:  # noqa: S112
                        continue
                elif isinstance(item, element_type):
                    validated_list.append(item)
                else:
                    continue
            else:
                if isinstance(item, element_type):
                    validated_list.append(item)
                else:
                    continue

        if not validated_list:
            return MissingValue

        return validated_list

    @classmethod
    def _process_nested_model_value(
        cls, value: t.Any, field_type: type
    ) -> BaseModel | t.Any:
        """
        Process a nested model value.

        Args:
            value: The value to process.
            field_type: The BaseModel subclass type.

        Returns:
            A BaseModel instance or MissingValue.
        """
        if isinstance(value, dict):
            return field_type.from_dict(value)  # type: ignore[attr-defined]
        if isinstance(value, field_type):
            return value
        return MissingValue

    @staticmethod
    def _process_primitive_value(value: t.Any, field_type: type) -> t.Any:
        """
        Process a primitive value.

        Args:
            value: The value to process.
            field_type: The expected type.

        Returns:
            The value if it matches the type, MissingValue otherwise.
        """
        if isinstance(value, field_type):
            return value
        return MissingValue

    def __init__(self, **kwargs: t.Any) -> None:
        """
        Initialize a BaseModel instance.

        Args:
            **kwargs: Field values to set. Fields not provided will have
                MissingValue. Fields with invalid types will also have MissingValue.
        """
        annotations = getattr(self.__class__, "__annotations__", {})
        fields: dict[str, Field | BaseModel] = {}

        for field_name, field_type in annotations.items():
            if field_name.startswith("_"):
                continue

            self._validate_field_type(field_type)

            value = kwargs.get(field_name, MissingValue)

            if self._is_list_type(field_type):
                element_type = self._get_list_element_type(field_type)

                element_origin = t.get_origin(element_type)
                if element_origin is t.Union or element_origin is types.UnionType:
                    raise UnsupportedListTypeError(element_type)

                self._validate_field_type(element_type)

                if value is not MissingValue:
                    value = self._process_list_value(value, element_type)

                fields[field_name] = Field(
                    name=field_name,
                    type=field_type,
                    value=value,
                    specs=None,
                )
                continue

            is_nested_model = isinstance(field_type, type) and issubclass(
                field_type, BaseModel
            )

            if value is not MissingValue:
                if is_nested_model:
                    value = self._process_nested_model_value(value, field_type)
                else:
                    value = self._process_primitive_value(value, field_type)

            if is_nested_model:
                if value is not MissingValue and isinstance(value, BaseModel):
                    fields[field_name] = value
                else:
                    fields[field_name] = Field(
                        name=field_name,
                        type=field_type,
                        value=value,
                        specs=None,
                    )
            else:
                fields[field_name] = Field(
                    name=field_name,
                    type=field_type,
                    value=value,
                    specs=None,
                )

        setattr(self, "_fields", fields)  # noqa: B010
        setattr(self, "_initialized", True)  # noqa: B010

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> t.Self:
        """
        Create a BaseModel instance from a dictionary.

        Args:
            data: Dictionary mapping field names to values.

        Returns:
            A new instance of the model with fields populated from the dictionary.
        """
        return cls(**data)

    @property
    def fields(self) -> FieldCollection:
        """
        Get the FieldCollection for this instance.

        Returns:
            The FieldCollection containing all fields.
        """
        return FieldCollection(self._fields)

    def __setattr__(self, name: str, value: t.Any) -> None:
        """
        Prevent setting attributes after initialization.

        Args:
            name: The attribute name.
            value: The value to set.

        Raises:
            AttributeError: If trying to set an attribute after initialization.
        """
        if hasattr(self, "_initialized") and self._initialized:
            raise AttributeError(
                f"'{self.__class__.__name__}' object attributes are readonly"
            )
        object.__setattr__(self, name, value)
