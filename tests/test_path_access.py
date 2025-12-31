import pytest

from cobjectric import (
    BaseModel,
    FillRateFieldResult,
    FillRateModelResult,
)


def test_path_access_simple() -> None:
    """Test that result['name'] works for simple field access."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    assert isinstance(result["name"], FillRateFieldResult)
    assert result["name"].value == 1.0
    assert result["age"].value == 1.0


def test_path_access_nested() -> None:
    """Test that result['address.city'] works for nested models."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {
            "name": "John",
            "address": {"street": "123 Main St", "city": "Anytown"},
        }
    )
    result = person.compute_fill_rate()

    assert isinstance(result["address"], FillRateModelResult)
    assert isinstance(result["address.city"], FillRateFieldResult)
    assert result["address.city"].value == 1.0
    assert result["address.street"].value == 1.0


def test_path_access_deeply_nested() -> None:
    """Test that result['address.country.name'] works for deeply nested models."""

    class Country(BaseModel):
        name: str
        code: str

    class Address(BaseModel):
        street: str
        country: Country

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {
            "name": "John",
            "address": {
                "street": "123 Main St",
                "country": {"name": "USA", "code": "US"},
            },
        }
    )
    result = person.compute_fill_rate()

    assert isinstance(result["address.country"], FillRateModelResult)
    assert isinstance(result["address.country.name"], FillRateFieldResult)
    assert result["address.country.name"].value == 1.0
    assert result["address.country.code"].value == 1.0


def test_path_access_invalid_raises_keyerror() -> None:
    """Test that accessing invalid path raises KeyError."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()

    with pytest.raises(KeyError):
        _ = result["non_existent"]

    with pytest.raises(KeyError):
        _ = result["name.invalid"]


def test_path_access_basemodel_list_index_raises() -> None:
    """Test that model['items[0]'] raises KeyError (not yet supported)."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
            ],
        }
    )

    # List index access is parsed but not yet supported
    with pytest.raises(KeyError, match="List index access not yet supported"):
        _ = order["items[0]"]


def test_path_access_basemodel_nested() -> None:
    """Test that model['address.city'] works for BaseModel."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {
            "name": "John",
            "address": {"street": "123 Main St", "city": "Anytown"},
        }
    )

    assert person["address"].fields.city.value == "Anytown"
    assert person["address.city"].value == "Anytown"
    assert person["address.street"].value == "123 Main St"


def test_path_access_basemodel_invalid_raises_keyerror() -> None:
    """Test that accessing invalid path on BaseModel raises KeyError."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    with pytest.raises(KeyError):
        _ = person["non_existent"]

    with pytest.raises(KeyError):
        _ = person["name.invalid"]


def test_path_access_fill_rate_result_nested_missing() -> None:
    """Test that path access works when nested model is missing."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person(name="John")  # address is missing
    result = person.compute_fill_rate()

    assert isinstance(result["address"], FillRateModelResult)
    assert result["address.street"].value == 0.0
    assert result["address.city"].value == 0.0


def test_path_access_fill_rate_result_list_index_raises() -> None:
    """Test that result['items[0]'] raises KeyError (not yet supported)."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana", "price": 0.5},
            ],
        }
    )
    result = order.compute_fill_rate()

    # List index access is parsed but not yet supported
    with pytest.raises(KeyError, match="List index access not yet supported"):
        _ = result["items[0]"]

    # Accessing list field without index works
    assert isinstance(result["items"], FillRateFieldResult)


def test_path_access_fill_rate_result_list_nested_raises() -> None:
    """Test that result['items[0].name'] raises KeyError (not yet supported)."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
            ],
        }
    )
    result = order.compute_fill_rate()

    # List index access is parsed but not yet supported
    with pytest.raises(KeyError, match="List index access not yet supported"):
        _ = result["items[0].name"]


def test_path_access_multiple_list_indices_raises() -> None:
    """Test that result['orders[0].items[1].name'] raises KeyError (not yet supported)."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    class Customer(BaseModel):
        orders: list[Order]

    customer = Customer.from_dict(
        {
            "orders": [
                {
                    "items": [
                        {"name": "Item 1"},
                        {"name": "Item 2"},
                    ],
                },
            ],
        }
    )
    result = customer.compute_fill_rate()

    # List index access is parsed but not yet supported
    with pytest.raises(KeyError, match="List index access not yet supported"):
        _ = result["orders[0].items[1].name"]


def test_path_access_invalid_bracket_syntax_unclosed() -> None:
    """Test that invalid bracket syntax raises KeyError."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    # Unclosed bracket
    with pytest.raises(KeyError, match="Invalid path"):
        _ = person["name[0"]


def test_path_access_invalid_bracket_syntax_non_numeric() -> None:
    """Test that non-numeric inside brackets raises KeyError."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    # Non-numeric index
    with pytest.raises(KeyError, match="Invalid path"):
        _ = person["name[abc]"]


def test_path_access_field_collection_empty_segments() -> None:
    """Test that empty segments list raises KeyError."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    # Empty segments should raise KeyError
    with pytest.raises(KeyError, match="Empty path"):
        _ = person.fields._resolve_path([])


def test_path_access_nested_non_model_field() -> None:
    """Test accessing nested path on a non-model field."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John", age=30)

    # Try to access nested path on non-model field
    with pytest.raises(KeyError, match="Cannot access"):
        _ = person["name.invalid"]


def test_path_access_fill_rate_result_nested_non_model() -> None:
    """Test accessing nested path on a non-model field in fill rate result."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    # Try to access nested path on non-model field
    with pytest.raises(KeyError, match="Cannot access"):
        _ = result["name.invalid"]


def test_path_access_current_none_result() -> None:
    """Test that accessing valid path but with invalid continuation returns the right error."""

    class Address(BaseModel):
        street: str

    class Person(BaseModel):
        name: str
        address: Address | None = None

    person = Person(name="John", address=None)

    # address is None, trying to access nested field should fail
    with pytest.raises(KeyError, match="Cannot access"):
        _ = person["address.street"]


def test_path_access_field_collection_direct_list_index() -> None:
    """Test FieldCollection with direct list index access."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    # Direct list index access
    with pytest.raises(KeyError, match="List index access not yet supported"):
        _ = person.fields._resolve_path(["[0]"])


def test_path_access_field_collection_non_model_non_field() -> None:
    """Test FieldCollection accessing nested on something that's neither BaseModel nor Field."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    # Manually insert a non-Field, non-BaseModel value
    person.fields._fields["name"] = "string_value"  # type: ignore

    # Try to access nested field
    with pytest.raises(KeyError, match="Cannot access"):
        _ = person.fields._resolve_path(["name", "nested"])


def test_path_access_nested_field_with_basemodel_value() -> None:
    """Test accessing nested path where Field contains a BaseModel."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {"name": "John", "address": {"street": "123 Main", "city": "Anytown"}}
    )

    # Access nested field through path
    street_field = person["address.street"]
    assert street_field.value == "123 Main"


def test_path_access_fill_rate_direct_list_index_first_segment() -> None:
    """Test FillRateFieldCollection with list index as first segment."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()

    # Direct list index access as first segment
    with pytest.raises(KeyError, match="List index access not yet supported"):
        _ = result.fields._resolve_path(["[0]"])


def test_path_access_field_collection_field_with_nested_basemodel() -> None:
    """Test FieldCollection path access through Field containing BaseModel value."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {"name": "John", "address": {"street": "123 Main", "city": "Anytown"}}
    )

    # Access deeply nested field - this goes through Field.value which is BaseModel
    city_field = person["address.city"]
    assert city_field.value == "Anytown"

    # Also test accessing through fields directly
    city_field_direct = person.fields._resolve_path(["address", "city"])
    assert city_field_direct.value == "Anytown"


def test_path_access_field_wrapping_basemodel() -> None:
    """Test path access when a Field wraps a BaseModel value (not MissingValue).

    This specifically tests line 159 in field_collection.py where:
    - current is a Field
    - current.value is not MissingValue
    - current.value is a BaseModel instance
    """
    from cobjectric.field import Field
    from cobjectric.field_spec import FieldSpec

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    # Create Person with address as a Field wrapping a BaseModel
    person = Person(name="John")
    address_instance = Address(street="123 Main", city="NYC")
    person._fields["address"] = Field(
        name="address",
        type=Address,
        value=address_instance,
        spec=FieldSpec(),
    )

    # Access nested field through the Field that wraps a BaseModel
    street_field = person.fields._resolve_path(["address", "street"])
    assert street_field.value == "123 Main"

    city_field = person.fields._resolve_path(["address", "city"])
    assert city_field.value == "NYC"
