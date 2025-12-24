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


def test_from_dict_basic() -> None:
    """Test that from_dict creates an instance from a dictionary."""

    class Person(BaseModel):
        name: str
        age: int
        email: str
        is_active: bool

    person = Person.from_dict(
        {
            "name": "John Doe",
            "age": 30,
            "email": "john.doe@example.com",
            "is_active": True,
        }
    )
    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value == 30
    assert person.fields.email.value == "john.doe@example.com"
    assert person.fields.is_active.value is True


def test_from_dict_missing_field() -> None:
    """Test that missing fields in dict result in MissingValue."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person = Person.from_dict({"name": "John Doe", "age": 30})
    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value == 30
    assert person.fields.email.value is MissingValue


def test_from_dict_invalid_type() -> None:
    """Test that invalid types in dict result in MissingValue."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person.from_dict({"name": "John Doe", "age": "invalid"})
    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value is MissingValue


def test_from_dict_empty_dict() -> None:
    """Test that from_dict works with an empty dictionary."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person.from_dict({})
    assert person.fields.name.value is MissingValue
    assert person.fields.age.value is MissingValue


def test_from_dict_extra_keys() -> None:
    """Test that extra keys in dict are ignored."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person.from_dict({"name": "John Doe", "age": 30, "extra_field": "ignored"})
    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value == 30
    with pytest.raises(AttributeError):
        _ = person.fields.extra_field


def test_nested_model_from_dict() -> None:
    """Test that nested models can be created from dict."""

    class Address(BaseModel):
        street: str
        city: str
        state: str
        zip_code: str
        country: str

    class Person(BaseModel):
        name: str
        age: int
        email: str
        is_active: bool
        address: Address

    person = Person.from_dict(
        {
            "name": "John Doe",
            "age": 30,
            "email": "john.doe@example.com",
            "is_active": True,
            "address": {
                "street": "123 Main St",
                "city": "Anytown",
                "state": "CA",
                "zip_code": "12345",
                "country": "USA",
            },
        }
    )
    assert isinstance(person.fields.address, Address)
    assert person.fields.address.fields.street.value == "123 Main St"
    assert person.fields.address.fields.city.value == "Anytown"
    assert person.fields.address.fields.state.value == "CA"
    assert person.fields.address.fields.zip_code.value == "12345"
    assert person.fields.address.fields.country.value == "USA"


def test_nested_model_access_via_fields() -> None:
    """Test that nested model fields can be accessed via .fields."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {
            "name": "John Doe",
            "address": {"street": "123 Main St", "city": "Anytown"},
        }
    )
    assert person.fields.name.value == "John Doe"
    assert isinstance(person.fields.address, Address)
    assert person.fields.address.fields.street.value == "123 Main St"
    assert person.fields.address.fields.city.value == "Anytown"


def test_nested_model_with_kwargs() -> None:
    """Test that nested models can be created with instance in kwargs."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    address = Address(street="123 Main St", city="Anytown")
    person = Person(name="John Doe", address=address)
    assert person.fields.name.value == "John Doe"
    assert isinstance(person.fields.address, Address)
    assert person.fields.address.fields.street.value == "123 Main St"
    assert person.fields.address.fields.city.value == "Anytown"


def test_nested_model_missing_value() -> None:
    """Test that missing nested model results in MissingValue."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict({"name": "John Doe"})
    assert person.fields.name.value == "John Doe"
    assert person.fields.address.value is MissingValue


def test_nested_model_invalid_type() -> None:
    """Test that invalid type for nested model results in MissingValue."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict({"name": "John Doe", "address": "invalid"})
    assert person.fields.name.value == "John Doe"
    assert person.fields.address.value is MissingValue


def test_deeply_nested_models() -> None:
    """Test that deeply nested models work correctly."""

    class Country(BaseModel):
        name: str
        code: str

    class Address(BaseModel):
        street: str
        city: str
        country: Country

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {
            "name": "John Doe",
            "address": {
                "street": "123 Main St",
                "city": "Anytown",
                "country": {"name": "United States", "code": "US"},
            },
        }
    )
    assert person.fields.name.value == "John Doe"
    assert isinstance(person.fields.address, Address)
    assert person.fields.address.fields.street.value == "123 Main St"
    assert isinstance(person.fields.address.fields.country, Country)
    assert person.fields.address.fields.country.fields.name.value == "United States"
    assert person.fields.address.fields.country.fields.code.value == "US"


def test_nested_model_with_invalid_dict() -> None:
    """Test nested model with dict that has invalid field types."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {
            "name": "John Doe",
            "address": {"street": "123 Main St", "city": 123},
        }
    )
    assert person.fields.name.value == "John Doe"
    assert isinstance(person.fields.address, Address)
    assert person.fields.address.fields.street.value == "123 Main St"
    assert person.fields.address.fields.city.value is MissingValue
