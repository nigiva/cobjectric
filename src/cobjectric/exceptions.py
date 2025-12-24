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
