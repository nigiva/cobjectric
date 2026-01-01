import pytest

from cobjectric import BaseModel
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


def test_field_collection_list_list_path() -> None:
    """Test path resolution with list[list[...]] structure."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    # Manually set a field to a list of lists
    from cobjectric.field import Field
    from cobjectric.field_spec import FieldSpec

    person.fields._fields["data"] = Field(
        name="data",
        type=list[list[str]],
        value=[["a", "b"], ["c", "d"]],
        spec=FieldSpec(),
    )

    # Access nested list item: data[0][1]
    item = person.fields._resolve_path(["data", "[0]", "[1]"])
    assert item == "b"


def test_field_collection_list_path_non_list_element() -> None:
    """Test path resolution when trying to index a non-list element."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    from cobjectric.field import Field
    from cobjectric.field_spec import FieldSpec

    person.fields._fields["items"] = Field(
        name="items",
        type=list[str],
        value=["a", "b", "c"],
        spec=FieldSpec(),
    )

    # Try to access items[0][0] - items[0] is "a" (str), not a list
    with pytest.raises(KeyError, match="Cannot use index on non-list element"):
        _ = person.fields._resolve_path(["items", "[0]", "[0]"])


def test_field_collection_list_path_non_model_element() -> None:
    """Test path resolution when trying to access field on non-model element."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    from cobjectric.field import Field
    from cobjectric.field_spec import FieldSpec

    person.fields._fields["items"] = Field(
        name="items",
        type=list[str],
        value=["a", "b"],
        spec=FieldSpec(),
    )

    # Try to access items[0].name - items[0] is "a" (str), not a BaseModel
    with pytest.raises(KeyError, match="Cannot access 'name' on non-model element"):
        _ = person.fields._resolve_path(["items", "[0]", "name"])


def test_field_collection_resolve_next_with_list() -> None:
    """Test _resolve_next when current is a list (list[list[...]] case)."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    from cobjectric.field import Field
    from cobjectric.field_spec import FieldSpec

    # Set up a field with list[list[str]]
    person.fields._fields["data"] = Field(
        name="data",
        type=list[list[str]],
        value=[["a", "b"], ["c", "d"]],
        spec=FieldSpec(),
    )

    # Access data[0] to get the first list
    first_list = person.fields._resolve_path(["data", "[0]"])
    assert first_list == ["a", "b"]

    # Now test _resolve_next directly with a list as current
    # This covers the elif isinstance(current, list) branch
    result = person.fields._resolve_next(first_list, "data[0]", ["[1]"])
    assert result == "b"


def test_field_collection_resolve_next_non_list_non_field() -> None:
    """Test _resolve_next when current is neither Field nor list but index is used."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    # Call _resolve_next with a string (neither Field nor list) and an index
    # This should raise KeyError
    with pytest.raises(KeyError, match="Cannot use index on non-list field"):
        person.fields._resolve_next("not a field or list", "segment", ["[0]"])
