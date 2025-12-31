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

    @classmethod
    def _validate_field_type(cls, field_type: type) -> None:
        """
        Validate that a field type is supported.

        Args:
            field_type: The type to validate.

        Raises:
            UnsupportedTypeError: If the type is not JSON-compatible.
            MissingListTypeArgError: If list is used without type arguments.
        """
        # Handle Union types - validate each type in the union
        if cls._is_union_type(field_type):
            union_args = cls._get_union_args(field_type)
            for single_type in union_args:
                cls._validate_field_type(single_type)
            return

        # Handle list types
        if field_type is list or field_type is t.List:
            args = t.get_args(field_type)
            if not args:
                raise MissingListTypeArgError()

        origin = t.get_origin(field_type)

        if origin is list:
            args = t.get_args(field_type)
            if not args:
                raise MissingListTypeArgError()
            # Validate element type recursively
            element_type = args[0]
            element_origin = t.get_origin(element_type)
            if element_origin is t.Union or element_origin is types.UnionType:
                raise UnsupportedListTypeError(element_type)
            cls._validate_field_type(element_type)

        # Handle dict types - validate key and value types recursively
        if cls._is_dict_type(field_type):
            args = t.get_args(field_type)
            if args:
                key_type, value_type = args
                cls._validate_field_type(key_type)
                cls._validate_field_type(value_type)
            return

        # Reject unsupported types
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

    @staticmethod
    def _is_union_type(field_type: type) -> bool:
        """
        Check if a field type is a Union type.

        Args:
            field_type: The type to check.

        Returns:
            True if the type is a Union, False otherwise.
        """
        origin = t.get_origin(field_type)
        return origin is t.Union or origin is types.UnionType

    @staticmethod
    def _get_union_args(field_type: type) -> tuple[type, ...]:
        """
        Extract the types from a Union type.

        Args:
            field_type: The Union type.

        Returns:
            Tuple of types in the union.
        """
        return t.get_args(field_type)

    @staticmethod
    def _is_dict_type(field_type: type) -> bool:
        """
        Check if a field type is a dict type.

        Args:
            field_type: The type to check.

        Returns:
            True if the type is a dict, False otherwise.
        """
        if field_type is dict:
            return True
        origin = t.get_origin(field_type)
        return origin is dict

    @staticmethod
    def _is_base_model_type(field_type: type) -> bool:
        """
        Check if a field type is a BaseModel subclass.

        Args:
            field_type: The type to check.

        Returns:
            True if the type is a BaseModel subclass, False otherwise.
        """
        return (
            isinstance(field_type, type)
            and issubclass(field_type, BaseModel)
            and field_type is not BaseModel
        )

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
        is_base_model = isinstance(field_type, type) and issubclass(
            field_type, BaseModel
        )

        if isinstance(value, dict) and is_base_model:
            assert issubclass(field_type, BaseModel)
            return field_type.from_dict(value)
        if isinstance(value, field_type):
            return value
        return MissingValue

    @classmethod
    def _validate_and_process_value(cls, value: t.Any, field_type: type) -> t.Any:
        """
        Recursively validate and process a value against a type.

        This is the central entry point for all type validation.
        It handles Union types, list types, dict types, BaseModel types,
        and primitive types recursively.

        Args:
            value: The value to validate.
            field_type: The expected type.

        Returns:
            The validated value (potentially filtered for collections),
            or MissingValue if completely invalid.
        """
        # 1. Handle Union types first
        if cls._is_union_type(field_type):
            return cls._process_union_value(value, field_type)

        # 2. Handle list types
        if cls._is_list_type(field_type):
            element_type = cls._get_list_element_type(field_type)
            return cls._process_list_value(value, element_type)

        # 3. Handle dict types (bare dict or dict[K, V])
        if cls._is_dict_type(field_type):
            return cls._process_dict_value(value, field_type)

        # 4. Handle BaseModel types
        if cls._is_base_model_type(field_type):
            return cls._process_nested_model_value(value, field_type)

        # 5. Handle primitives and NoneType
        return cls._process_primitive_value(value, field_type)

    @classmethod
    def _process_union_value(cls, value: t.Any, field_type: type) -> t.Any:
        """
        Process a union type value.

        Tries each type in order and returns the first match.

        Args:
            value: The value to process.
            field_type: The Union type.

        Returns:
            The validated value or MissingValue if no type matches.
        """
        union_args = cls._get_union_args(field_type)

        for single_type in union_args:
            result = cls._validate_and_process_value(value, single_type)
            if result is not MissingValue:
                return result

        return MissingValue

    @classmethod
    def _process_dict_value(cls, value: t.Any, field_type: type) -> t.Any:
        """
        Process a dict value with recursive type validation and partial filtering.

        Args:
            value: The value to process.
            field_type: The dict type (dict or dict[K, V]).

        Returns:
            A validated dict with filtered invalid entries,
            or MissingValue if value is not a dict or all entries are invalid.
        """
        if not isinstance(value, dict):
            return MissingValue

        # Bare dict type - no validation needed
        if field_type is dict:
            return value

        args = t.get_args(field_type)
        if not args:
            return value

        key_type, value_type = args
        validated_dict: dict[t.Any, t.Any] = {}
        original_empty = len(value) == 0

        for k, v in value.items():
            # Validate key (keys are not filtered, they must match exactly)
            validated_k = cls._validate_and_process_value(k, key_type)
            if validated_k is MissingValue:
                continue  # Skip this pair

            # Recursively validate value
            validated_v = cls._validate_and_process_value(v, value_type)
            if validated_v is not MissingValue:
                validated_dict[validated_k] = validated_v

        # Empty dict is valid, return it even if empty
        if original_empty:
            return validated_dict

        # Non-empty dict: return validated dict if it has entries
        return validated_dict if validated_dict else MissingValue

    @classmethod
    def _process_list_value(
        cls, value: t.Any, element_type: type
    ) -> list[t.Any] | t.Any:
        """
        Process a list value with recursive element validation.

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

        validated_list = []
        for item in value:
            validated_item = cls._validate_and_process_value(item, element_type)
            if validated_item is not MissingValue:
                validated_list.append(validated_item)

        return validated_list if validated_list else MissingValue

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
        # Handle NoneType
        if field_type is type(None):
            return value if value is None else MissingValue

        # Handle primitives
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

            if value is not MissingValue:
                value = self._validate_and_process_value(value, field_type)

            # Check if this is a nested model
            is_nested_model = self._is_base_model_type(field_type)

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
