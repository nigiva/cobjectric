from cobjectric import FieldSpec, MissingValue
from cobjectric.field import Field


def test_field_creation() -> None:
    """Test that Field can be created with all attributes."""
    field = Field(
        name="test_field",
        type=str,
        value="test_value",
        spec=FieldSpec(),
    )
    assert field.name == "test_field"
    assert field.type is str
    assert field.value == "test_value"
    assert isinstance(field.spec, FieldSpec)


def test_field_with_missing_value() -> None:
    """Test that Field can have MissingValue as value."""
    field = Field(
        name="test_field",
        type=str,
        value=MissingValue,
        spec=FieldSpec(),
    )
    assert field.value is MissingValue


def test_field_repr() -> None:
    """Test that Field has a proper representation."""
    field = Field(
        name="test_field",
        type=str,
        value="test_value",
        spec=FieldSpec(),
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
        spec=FieldSpec(),
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
        spec=FieldSpec(),
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
        spec=FieldSpec(),
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
        spec=FieldSpec(),
    )
    repr_str = repr(field)
    assert "test_field" in repr_str
    assert "test_value" in repr_str
    # Should not raise AttributeError


def test_field_spec_default_value() -> None:
    """Test that Field has a default FieldSpec when spec is not provided."""
    field = Field(
        name="test_field",
        type=str,
        value="test_value",
    )
    assert isinstance(field.spec, FieldSpec)
    assert field.spec.metadata == {}


def test_field_spec_default_value_each_instance_has_own_spec() -> None:
    """Test that each Field instance gets its own FieldSpec by default."""
    field1 = Field(name="field1", type=str, value="value1")
    field2 = Field(name="field2", type=int, value=42)

    assert isinstance(field1.spec, FieldSpec)
    assert isinstance(field2.spec, FieldSpec)
    # Each instance should have its own FieldSpec
    assert field1.spec is not field2.spec
    # Modifying one should not affect the other
    field1.spec.metadata["key"] = "value"
    assert field1.spec.metadata == {"key": "value"}
    assert field2.spec.metadata == {}
