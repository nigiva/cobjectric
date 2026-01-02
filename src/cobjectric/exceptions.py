import typing as t


class CobjectricError(Exception):
    """Base exception for all cobjectric errors."""


class UnsupportedListTypeError(CobjectricError):
    """
    Exception raised when a list field has an unsupported type.

    This exception is raised when a list field contains a Union type,
    which is not supported. Only single types are allowed in lists.
    """

    def __init__(self, unsupported_type: type) -> None:
        """
        Initialize UnsupportedListTypeError.

        Args:
            unsupported_type: The unsupported type that was detected.
        """
        self.unsupported_type = unsupported_type
        type_name = getattr(unsupported_type, "__name__", str(unsupported_type))
        super().__init__(
            f"Unsupported list type: list[{type_name}]. "
            "List fields must contain a single type (e.g., list[str], list[int], "
            "list[MyModel]). Union types like list[str | int] are not supported."
        )


class MissingListTypeArgError(CobjectricError):
    """
    Exception raised when a list type is used without type arguments.

    This exception is raised when using bare 'list' or 't.List' without
    specifying the element type.
    """

    def __init__(self) -> None:
        """Initialize MissingListTypeArgError."""
        super().__init__(
            "List type must specify an element type. "
            "Use list[str], list[int], list[MyModel], etc. instead of bare 'list'."
        )


class UnsupportedTypeError(CobjectricError):
    """
    Exception raised when a field type is not supported.

    This exception is raised when a field type is not JSON-compatible.
    Only str, int, float, bool, list[T], or BaseModel subclasses are allowed.
    """

    def __init__(self, unsupported_type: type) -> None:
        """
        Initialize UnsupportedTypeError.

        Args:
            unsupported_type: The unsupported type that was detected.
        """
        self.unsupported_type = unsupported_type
        type_name = getattr(unsupported_type, "__name__", str(unsupported_type))
        super().__init__(
            f"Unsupported type: {type_name}. "
            "Only JSON-compatible types are allowed: str, int, float, bool, "
            "list[T], or BaseModel subclasses."
        )


class DuplicateFillRateFuncError(CobjectricError):
    """
    Exception raised when multiple fill_rate_func are defined for the same field.

    This exception is raised when a field has both a Spec(fill_rate_func=...)
    and a @fill_rate_func decorator, or multiple @fill_rate_func decorators.
    """

    def __init__(self, field_name: str) -> None:
        """
        Initialize DuplicateFillRateFuncError.

        Args:
            field_name: The name of the field with duplicate fill_rate_func.
        """
        self.field_name = field_name
        super().__init__(
            f"Multiple fill_rate_func defined for field '{field_name}'. "
            "A field can only have one fill_rate_func (either from Spec() or "
            "@fill_rate_func decorator, not both)."
        )


class InvalidFillRateValueError(CobjectricError):
    """
    Exception raised when fill_rate_func returns an invalid value.

    This exception is raised when fill_rate_func returns a value that is not
    a float (or int convertible to float) or is not in the range [0, 1].
    """

    def __init__(self, field_name: str, value: t.Any) -> None:
        """
        Initialize InvalidFillRateValueError.

        Args:
            field_name: The name of the field with invalid fill_rate value.
            value: The invalid value that was returned.
        """
        self.field_name = field_name
        self.value = value
        value_type = type(value).__name__
        super().__init__(
            f"Invalid fill_rate value for field '{field_name}': {value!r} "
            f"(type: {value_type}). Fill rate must be a float between 0.0 and 1.0."
        )


class InvalidWeightError(CobjectricError):
    """
    Exception raised when weight is invalid.

    This exception is raised when weight is negative (< 0.0).
    Weight must be >= 0.0.
    """

    def __init__(
        self, weight: float, source: str, weight_type: str = "fill_rate"
    ) -> None:
        """
        Initialize InvalidWeightError.

        Args:
            weight: The invalid weight value.
            source: The source of the weight ("Spec" or "decorator").
            weight_type: The type of weight ("fill_rate" or "fill_rate_accuracy").
        """
        self.weight = weight
        self.source = source
        self.weight_type = weight_type
        if weight_type == "fill_rate":
            super().__init__(
                f"Invalid weight in {source}: {weight}. Weight must be >= 0.0."
            )
        else:
            super().__init__(
                f"Invalid {weight_type}_weight in {source}: {weight}. "
                "Weight must be >= 0.0."
            )


class DuplicateFillRateAccuracyFuncError(CobjectricError):
    """
    Exception raised when multiple fill_rate_accuracy_func are defined.

    This exception is raised when a field has both a
    Spec(fill_rate_accuracy_func=...) and a @fill_rate_accuracy_func decorator,
    or multiple @fill_rate_accuracy_func decorators.
    """

    def __init__(self, field_name: str) -> None:
        """
        Initialize DuplicateFillRateAccuracyFuncError.

        Args:
            field_name: The name of the field with duplicate fill_rate_accuracy_func.
        """
        self.field_name = field_name
        super().__init__(
            f"Multiple fill_rate_accuracy_func defined for field '{field_name}'. "
            "A field can only have one fill_rate_accuracy_func (either from Spec() or "
            "@fill_rate_accuracy_func decorator, not both)."
        )


class DuplicateSimilarityFuncError(CobjectricError):
    """
    Exception raised when multiple similarity_func are defined.

    This exception is raised when a field has both a
    Spec(similarity_func=...) and a @similarity_func decorator,
    or multiple @similarity_func decorators.
    """

    def __init__(self, field_name: str) -> None:
        """
        Initialize DuplicateSimilarityFuncError.

        Args:
            field_name: The name of the field with duplicate similarity_func.
        """
        self.field_name = field_name
        super().__init__(
            f"Multiple similarity_func defined for field '{field_name}'. "
            "A field can only have one similarity_func (either from Spec() or "
            "@similarity_func decorator, not both)."
        )


class InvalidAggregatedFieldError(CobjectricError):
    """
    Exception raised when accessing an invalid field in aggregated_fields.

    This exception is raised when trying to access a field that doesn't exist
    in the aggregated model through aggregated_fields property.
    """

    def __init__(
        self,
        field_name: str,
        available_fields: list[str],
        model_type: type | None = None,
    ) -> None:
        """
        Initialize InvalidAggregatedFieldError.

        Args:
            field_name: The name of the field that was accessed.
            available_fields: List of available field names.
            model_type: The model type (optional) for additional context.
        """
        self.field_name = field_name
        self.available_fields = available_fields
        self.model_type = model_type
        fields_str = ", ".join(repr(f) for f in available_fields)
        type_str = f" (from {model_type.__name__})" if model_type else ""
        super().__init__(
            f"Invalid aggregated field '{field_name}'{type_str}. "
            f"Available fields: [{fields_str}]"
        )


class InvalidListCompareStrategyError(CobjectricError):
    """
    Exception raised when list_compare_strategy is used on a non-list field.

    This exception is raised when trying to use list_compare_strategy on a field
    that is not a list[BaseModel] type.
    """

    def __init__(self, field_name: str) -> None:
        """
        Initialize InvalidListCompareStrategyError.

        Args:
            field_name: The name of the field with invalid list_compare_strategy.
        """
        self.field_name = field_name
        super().__init__(
            f"list_compare_strategy can only be used on list[BaseModel] fields. "
            f"Field '{field_name}' is not a list[BaseModel] type."
        )


class IncompatibleModelResultError(CobjectricError):
    """
    Exception raised when trying to combine ModelResults from different model types.

    This exception is raised when trying to add ModelResults that come from
    different BaseModel classes. All ModelResults in a ModelResultCollection
    must come from the same model type.
    """

    def __init__(self, model_type1: type, model_type2: type) -> None:
        """
        Initialize IncompatibleModelResultError.

        Args:
            model_type1: The first model type.
            model_type2: The second model type.
        """
        self.model_type1 = model_type1
        self.model_type2 = model_type2
        type1_name = getattr(model_type1, "__name__", str(model_type1))
        type2_name = getattr(model_type2, "__name__", str(model_type2))
        super().__init__(
            f"Cannot combine ModelResults from different model types: "
            f"{type1_name} and {type2_name}. "
            "All ModelResults in a collection must come from the same model type."
        )
