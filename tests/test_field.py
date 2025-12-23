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
