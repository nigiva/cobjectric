import pytest

from cobjectric.field import Field
from cobjectric.field_collection import FieldCollection
from cobjectric.field_spec import FieldSpec


def test_field_collection_creation() -> None:
    """Test that FieldCollection can be created."""
    fields = {
        "name": Field(
            name="name",
            type=str,
            value="John",
            spec=FieldSpec(),
        ),
    }
    collection = FieldCollection(fields)
    assert collection is not None


def test_field_collection_access_by_attribute() -> None:
    """Test that FieldCollection allows access by attribute."""
    fields = {
        "name": Field(
            name="name",
            type=str,
            value="John",
            spec=FieldSpec(),
        ),
        "age": Field(
            name="age",
            type=int,
            value=30,
            spec=FieldSpec(),
        ),
    }
    collection = FieldCollection(fields)
    assert collection.name.value == "John"
    assert collection.age.value == 30


def test_field_collection_access_nonexistent_field() -> None:
    """Test that FieldCollection raises AttributeError for nonexistent field."""
    fields = {
        "name": Field(
            name="name",
            type=str,
            value="John",
            spec=FieldSpec(),
        ),
    }
    collection = FieldCollection(fields)
    with pytest.raises(AttributeError):
        _ = collection.nonexistent


def test_field_collection_iteration() -> None:
    """Test that FieldCollection can be iterated."""
    fields = {
        "name": Field(
            name="name",
            type=str,
            value="John",
            spec=FieldSpec(),
        ),
        "age": Field(
            name="age",
            type=int,
            value=30,
            spec=FieldSpec(),
        ),
    }
    collection = FieldCollection(fields)
    field_names = [field.name for field in collection]
    assert "name" in field_names
    assert "age" in field_names
    assert len(field_names) == 2


def test_field_collection_repr() -> None:
    """Test that FieldCollection has a proper representation."""
    fields = {
        "name": Field(
            name="name",
            type=str,
            value="John",
            spec=FieldSpec(),
        ),
    }
    collection = FieldCollection(fields)
    repr_str = repr(collection)
    assert "FieldCollection" in repr_str
    assert "name" in repr_str
