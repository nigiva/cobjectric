import typing as t

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

    _fields: dict[str, Field]
    _initialized: bool

    def __init__(self, **kwargs: t.Any) -> None:
        """
        Initialize a BaseModel instance.

        Args:
            **kwargs: Field values to set. Fields not provided will have
                MissingValue. Fields with invalid types will also have MissingValue.
        """
        annotations = self.__class__.__annotations__
        fields: dict[str, Field] = {}

        for field_name, field_type in annotations.items():
            if field_name.startswith("_"):
                continue

            value = kwargs.get(field_name, MissingValue)

            if value is not MissingValue:
                if not isinstance(value, field_type):
                    value = MissingValue

            fields[field_name] = Field(
                name=field_name,
                type=field_type,
                value=value,
                specs=None,
            )

        setattr(self, "_fields", fields)  # noqa: B010
        setattr(self, "_initialized", True)  # noqa: B010

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
