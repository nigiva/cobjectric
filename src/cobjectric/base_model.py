from __future__ import annotations

import fnmatch
import types
import typing as t

from cobjectric.exceptions import (
    DuplicateFillRateFuncError,
    InvalidFillRateValueError,
    MissingListTypeArgError,
    UnsupportedListTypeError,
    UnsupportedTypeError,
)
from cobjectric.field import Field
from cobjectric.field_collection import FieldCollection
from cobjectric.field_spec import FieldSpec
from cobjectric.fill_rate import (
    FillRateFieldResult,
    FillRateFunc,
    FillRateFuncInfo,
    FillRateModelResult,
)
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

    @classmethod
    def _collect_field_normalizers(
        cls,
    ) -> dict[str, list[t.Callable[..., t.Any]]]:
        """
        Collect all @field_normalizer decorated methods for each field.

        Returns:
            Dict mapping field_name to list of normalizer functions
        """
        normalizers: dict[str, list[t.Callable[..., t.Any]]] = {}
        annotations = getattr(cls, "__annotations__", {})
        field_names = [n for n in annotations if not n.startswith("_")]

        # Use __dict__ to preserve declaration order
        for attr_name, attr_value in cls.__dict__.items():
            if attr_name.startswith("_"):
                continue
            if callable(attr_value) and hasattr(attr_value, "_field_normalizers"):
                for info in attr_value._field_normalizers:
                    for pattern in info.field_patterns:
                        for field_name in field_names:
                            if fnmatch.fnmatch(field_name, pattern):
                                if field_name not in normalizers:
                                    normalizers[field_name] = []
                                normalizers[field_name].append(info.func)
        return normalizers

    @staticmethod
    def _build_combined_normalizer(
        spec_normalizer: t.Callable[..., t.Any] | None,
        decorator_normalizers: list[t.Callable[..., t.Any]],
    ) -> t.Callable[..., t.Any] | None:
        """
        Build a single normalizer from Spec + decorator normalizers.

        Args:
            spec_normalizer: Normalizer from Spec() if any.
            decorator_normalizers: List of normalizers from
                @field_normalizer decorators.

        Returns:
            Combined normalizer function or None if no normalizers.
        """
        all_normalizers: list[t.Callable[..., t.Any]] = []
        if spec_normalizer:
            all_normalizers.append(spec_normalizer)
        all_normalizers.extend(decorator_normalizers)

        if not all_normalizers:
            return None

        def combined(value: t.Any) -> t.Any:
            result = value
            for norm_func in all_normalizers:
                result = norm_func(result)
            return result

        return combined

    @classmethod
    def _collect_fill_rate_funcs(
        cls,
    ) -> dict[str, list[FillRateFuncInfo]]:
        """
        Collect all @fill_rate_func decorated methods for each field.

        Returns:
            Dict mapping field_name to list of FillRateFuncInfo
        """
        fill_rate_funcs: dict[str, list[FillRateFuncInfo]] = {}
        annotations = getattr(cls, "__annotations__", {})
        field_names = [n for n in annotations if not n.startswith("_")]

        # Use __dict__ to preserve declaration order
        for attr_name, attr_value in cls.__dict__.items():
            if attr_name.startswith("_"):
                continue
            if callable(attr_value) and hasattr(attr_value, "_fill_rate_funcs"):
                for info in attr_value._fill_rate_funcs:
                    for pattern in info.field_patterns:
                        for field_name in field_names:
                            if fnmatch.fnmatch(field_name, pattern):
                                if field_name not in fill_rate_funcs:
                                    fill_rate_funcs[field_name] = []
                                fill_rate_funcs[field_name].append(info)
        return fill_rate_funcs

    @staticmethod
    def _default_fill_rate_func(value: t.Any) -> float:
        """
        Default fill_rate_func: returns 0.0 if MissingValue, else 1.0.

        Args:
            value: The field value.

        Returns:
            float: 0.0 if MissingValue, 1.0 otherwise.
        """
        return 0.0 if value is MissingValue else 1.0

    @staticmethod
    def _validate_fill_rate_value(field_name: str, value: t.Any) -> float:
        """
        Validate that fill_rate_func returns a valid float in [0, 1].

        Args:
            field_name: The name of the field.
            value: The value returned by fill_rate_func.

        Returns:
            float: The validated fill rate value.

        Raises:
            InvalidFillRateValueError: If value is not a float or not in [0, 1].
        """
        # Accept int (0, 1) and convert to float
        if isinstance(value, int):
            value = float(value)

        if not isinstance(value, float):
            raise InvalidFillRateValueError(field_name, value)

        if not 0.0 <= value <= 1.0:
            raise InvalidFillRateValueError(field_name, value)

        return value

    def compute_fill_rate(self) -> FillRateModelResult:
        """
        Compute fill rate for all fields in this model.

        Returns:
            FillRateModelResult containing fill rates for all fields.

        Raises:
            DuplicateFillRateFuncError: If multiple fill_rate_func are defined
                for the same field.
            InvalidFillRateValueError: If a fill_rate_func returns an invalid value.
        """
        # Collect decorator fill_rate_funcs once per class
        decorator_fill_rate_funcs = self._collect_fill_rate_funcs()

        result_fields: dict[str, FillRateFieldResult | FillRateModelResult] = {}

        for field_name, field in self._fields.items():
            # Get FieldSpec
            if isinstance(field, Field):
                spec = field.spec
            else:
                # Nested model - use default spec
                spec = FieldSpec()

            # Check for duplicates
            spec_func = spec.fill_rate_func
            decorator_funcs = decorator_fill_rate_funcs.get(field_name, [])

            if spec_func and decorator_funcs:
                raise DuplicateFillRateFuncError(field_name)

            if len(decorator_funcs) > 1:
                raise DuplicateFillRateFuncError(field_name)

            # Get the fill_rate_func to use
            fill_rate_func_to_use: FillRateFunc | None = None
            if spec_func:
                fill_rate_func_to_use = spec_func
            elif decorator_funcs:
                fill_rate_func_to_use = decorator_funcs[0].func
            else:
                fill_rate_func_to_use = self._default_fill_rate_func

            # Get the weight to use (decorator > Spec > default 1.0)
            weight_to_use: float = 1.0
            if decorator_funcs:
                # Decorator weight takes precedence
                weight_to_use = decorator_funcs[0].weight
            elif spec.weight != 1.0:
                # Use Spec weight if different from default
                weight_to_use = spec.weight

            # Check if this is a nested model
            if isinstance(field, BaseModel):
                # Recursively compute fill rate for nested model
                nested_result = field.compute_fill_rate()
                result_fields[field_name] = nested_result
            else:
                # Check if this field is a nested model type but MissingValue
                is_nested_model_type = self._is_base_model_type(field.type)
                if is_nested_model_type and field.value is MissingValue:
                    # Create a FillRateModelResult with all fields at 0.0
                    nested_model_class = field.type
                    nested_annotations = getattr(
                        nested_model_class, "__annotations__", {}
                    )
                    nested_fields: dict[
                        str, FillRateFieldResult | FillRateModelResult
                    ] = {}
                    for nested_field_name in nested_annotations:
                        if nested_field_name.startswith("_"):
                            continue
                        nested_fields[nested_field_name] = FillRateFieldResult(
                            value=0.0, weight=1.0
                        )
                    result_fields[field_name] = FillRateModelResult(
                        _fields=nested_fields
                    )
                else:
                    # Compute fill rate for this field
                    field_value = field.value
                    fill_rate_value = fill_rate_func_to_use(field_value)
                    validated_value = self._validate_fill_rate_value(
                        field_name, fill_rate_value
                    )
                    result_fields[field_name] = FillRateFieldResult(
                        value=validated_value, weight=weight_to_use
                    )

        return FillRateModelResult(_fields=result_fields)

    def __init__(self, **kwargs: t.Any) -> None:
        """
        Initialize a BaseModel instance.

        Args:
            **kwargs: Field values to set. Fields not provided will have
                MissingValue. Fields with invalid types will also have MissingValue.
        """
        annotations = getattr(self.__class__, "__annotations__", {})
        fields: dict[str, Field | BaseModel] = {}

        # Collect decorator normalizers once per class
        field_normalizers = self._collect_field_normalizers()

        for field_name, field_type in annotations.items():
            if field_name.startswith("_"):
                continue

            self._validate_field_type(field_type)

            value = kwargs.get(field_name, MissingValue)

            # Get FieldSpec from class attribute (if Spec() was used)
            class_default = getattr(self.__class__, field_name, None)
            if isinstance(class_default, FieldSpec):
                spec = class_default
            else:
                spec = FieldSpec()  # Default

            # Apply normalizers before type validation
            if value is not MissingValue:
                combined_normalizer = self._build_combined_normalizer(
                    spec.normalizer,
                    field_normalizers.get(field_name, []),
                )
                if combined_normalizer:
                    value = combined_normalizer(value)
                    # Create new FieldSpec with combined normalizer,
                    # preserving fill_rate_func
                    spec = FieldSpec(
                        metadata=spec.metadata,
                        normalizer=combined_normalizer,
                        fill_rate_func=spec.fill_rate_func,
                    )

                # Then validate type
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
                        spec=spec,
                    )
            else:
                fields[field_name] = Field(
                    name=field_name,
                    type=field_type,
                    value=value,
                    spec=spec,
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
