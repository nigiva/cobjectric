import pytest

from cobjectric import BaseModel, MissingValue


def test_base_model_creation_with_kwargs() -> None:
    """Test that BaseModel can be created with kwargs."""

    class Person(BaseModel):
        name: str
        age: int
        email: str
        is_active: bool

    person = Person(
        name="John Doe",
        age=30,
        email="john.doe@example.com",
        is_active=True,
    )
    assert person is not None


def test_base_model_fields_access() -> None:
    """Test that BaseModel fields can be accessed via .fields."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John Doe", age=30)
    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value == 30


def test_base_model_fields_are_readonly() -> None:
    """Test that BaseModel fields cannot be modified after creation."""

    class Person(BaseModel):
        name: str

    person = Person(name="John Doe")
    with pytest.raises(AttributeError):
        person.name = "Jane Doe"


def test_base_model_missing_field_has_missing_value() -> None:
    """Test that missing fields have MissingValue."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John Doe")
    assert person.fields.age.value is MissingValue


def test_base_model_invalid_type_has_missing_value() -> None:
    """Test that fields with invalid type have MissingValue."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John Doe", age="invalid")
    assert person.fields.age.value is MissingValue


def test_base_model_valid_types() -> None:
    """Test that BaseModel accepts valid types."""

    class Person(BaseModel):
        name: str
        age: int
        is_active: bool

    person = Person(
        name="John Doe",
        age=30,
        is_active=True,
    )
    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value == 30
    assert person.fields.is_active.value is True


def test_base_model_multiple_instances() -> None:
    """Test that multiple BaseModel instances work independently."""

    class Person(BaseModel):
        name: str
        age: int

    person1 = Person(name="John", age=30)
    person2 = Person(name="Jane", age=25)
    assert person1.fields.name.value == "John"
    assert person2.fields.name.value == "Jane"


def test_base_model_ignores_private_fields() -> None:
    """Test that BaseModel ignores fields starting with underscore."""

    class Person(BaseModel):
        name: str
        _private_field: str

    person = Person(name="John")
    assert person.fields.name.value == "John"
    with pytest.raises(AttributeError):
        _ = person.fields._private_field


def test_base_model_without_annotations() -> None:
    """Test that BaseModel can be instantiated without field annotations."""

    class EmptyModel(BaseModel):
        pass

    instance = EmptyModel()
    assert instance is not None
    assert len(list(instance.fields)) == 0


def test_base_model_without_annotations_with_kwargs() -> None:
    """Test that BaseModel without annotations ignores kwargs."""

    class EmptyModel(BaseModel):
        pass

    instance = EmptyModel(extra_field="value")
    assert instance is not None
    assert len(list(instance.fields)) == 0
