from cobjectric import MissingValue
from cobjectric.field import Field


def test_field_creation() -> None:
    """Test that Field can be created with all attributes."""
    field = Field(
        name="test_field",
        type=str,
        value="test_value",
        specs=None,
    )
    assert field.name == "test_field"
    assert field.type is str
    assert field.value == "test_value"
    assert field.specs is None


def test_field_with_missing_value() -> None:
    """Test that Field can have MissingValue as value."""
    field = Field(
        name="test_field",
        type=str,
        value=MissingValue,
        specs=None,
    )
    assert field.value is MissingValue


def test_field_repr() -> None:
    """Test that Field has a proper representation."""
    field = Field(
        name="test_field",
        type=str,
        value="test_value",
        specs=None,
    )
    repr_str = repr(field)
    assert "test_field" in repr_str
    assert "str" in repr_str
    assert "test_value" in repr_str


def test_field_repr_with_generic_type() -> None:
    """Test that Field repr works with generic types like list[str]."""
    field = Field(
        name="tags",
        type=list[str],
        value=["tag1", "tag2"],
        specs=None,
    )
    repr_str = repr(field)
    assert "tags" in repr_str
    assert "list" in repr_str or "List" in repr_str
    assert "tag1" in repr_str or "tag2" in repr_str


def test_field_repr_with_dict_type() -> None:
    """Test that Field repr works with dict types like dict[str, int]."""
    field = Field(
        name="metadata",
        type=dict[str, int],
        value={"key": 42},
        specs=None,
    )
    repr_str = repr(field)
    assert "metadata" in repr_str
    assert "dict" in repr_str or "Dict" in repr_str


def test_field_repr_with_optional_type() -> None:
    """Test that Field repr works with Optional types."""
    field = Field(
        name="optional_field",
        type=str | None,
        value=None,
        specs=None,
    )
    repr_str = repr(field)
    assert "optional_field" in repr_str


def test_field_repr_with_type_without_name() -> None:
    """Test that Field repr works with types that don't have __name__ attribute."""

    class TypeWithoutName:
        def __repr__(self) -> str:
            return "CustomType"

    field = Field(
        name="test_field",
        type=TypeWithoutName(),  # Instance without __name__
        value="test_value",
        specs=None,
    )
    repr_str = repr(field)
    assert "test_field" in repr_str
    assert "test_value" in repr_str
    # Should not raise AttributeError
