import pytest

from cobjectric import (
    BaseModel,
    FillRateFieldResult,
    FillRateListResult,
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


def test_path_access_basemodel_list_index_works() -> None:
    """Test that model['items[0]'] works."""

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

    # List index access now works
    item = order["items[0]"]
    assert isinstance(item, BaseModel)
    assert item.fields.name.value == "Apple"


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


def test_path_access_fill_rate_result_list_index_works() -> None:
    """Test that result['items[0]'] works."""

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

    # List index access now works
    item_result = result["items[0]"]
    assert isinstance(item_result, FillRateModelResult)
    assert item_result.fields.name.value == 1.0

    # Accessing list field without index works
    assert isinstance(result["items"], FillRateListResult)


def test_path_access_fill_rate_result_list_nested_works() -> None:
    """Test that result['items[0].name'] works."""

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

    # List index access with nested field now works
    name_result = result["items[0].name"]
    assert isinstance(name_result, FillRateFieldResult)
    assert name_result.value == 1.0


def test_path_access_multiple_list_indices_works() -> None:
    """Test that result['orders[0].items[1].name'] works."""

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

    # Multiple list indices now work
    name_result = result["orders[0].items[1].name"]
    assert isinstance(name_result, FillRateFieldResult)
    assert name_result.value == 1.0


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

    # Direct list index access without field name should fail
    with pytest.raises(KeyError, match="Cannot use index on non-list"):
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

    # Direct list index access without field name should fail
    with pytest.raises(KeyError, match="Cannot use index on non-list"):
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


def test_path_access_list_index_works() -> None:
    """Test that path access with list index works: result['items[0]']."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
            ],
        }
    )
    result = order.compute_fill_rate()

    item0_result = result["items[0]"]
    assert isinstance(item0_result, FillRateModelResult)
    assert item0_result.fields.name.value == 1.0


def test_path_access_list_index_nested_field() -> None:
    """Test that path access with list index and nested field works: result['items[0].name']."""

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

    name_result = result["items[0].name"]
    assert isinstance(name_result, FillRateFieldResult)
    assert name_result.value == 1.0

    price_result = result["items[1].price"]
    assert isinstance(price_result, FillRateFieldResult)
    assert price_result.value == 1.0


def test_path_access_list_index_out_of_bounds() -> None:
    """Test that path access with out-of-bounds index raises KeyError."""

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
    result = order.compute_fill_rate()

    with pytest.raises(KeyError, match="List index.*out of range"):
        _ = result["items[99]"]


def test_path_access_list_multiple_indices() -> None:
    """Test that path access with multiple list indices works: orders[0].items[1].name."""

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
                        {"name": "Apple"},
                        {"name": "Banana"},
                    ],
                },
            ],
        }
    )
    result = customer.compute_fill_rate()

    name_result = result["orders[0].items[1].name"]
    assert isinstance(name_result, FillRateFieldResult)
    assert name_result.value == 1.0


def test_path_access_basemodel_list_index() -> None:
    """Test that path access with list index works on BaseModel: model['items[0].name']."""

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

    name_field = order["items[0].name"]
    assert name_field.value == "Apple"

    price_field = order["items[0].price"]
    assert price_field.value == 1.0


def test_path_access_invalid_list_index() -> None:
    """Test that invalid list index raises KeyError."""

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

    # Invalid index (non-numeric) - error is raised during parsing
    with pytest.raises(KeyError, match="Invalid path"):
        _ = order["items[abc]"]


def test_path_access_list_index_out_of_range() -> None:
    """Test that out-of-range list index raises KeyError."""

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

    # Out of range index
    with pytest.raises(KeyError, match="List index.*out of range"):
        _ = order["items[99]"]


def test_path_access_non_list_field_with_index() -> None:
    """Test that using index on non-list field raises KeyError."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    # Try to use index on non-list field
    with pytest.raises(KeyError, match="Cannot use index on non-list"):
        _ = person["name[0]"]


def test_path_access_fill_rate_invalid_list_index() -> None:
    """Test that invalid list index in fill rate result raises KeyError."""

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
    result = order.compute_fill_rate()

    # Invalid index (non-numeric) - error is raised during parsing
    with pytest.raises(KeyError, match="Invalid path"):
        _ = result["items[abc]"]


def test_path_access_fill_rate_list_index_out_of_range() -> None:
    """Test that out-of-range list index in fill rate result raises KeyError."""

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
    result = order.compute_fill_rate()

    # Out of range index
    with pytest.raises(KeyError, match="List index.*out of range"):
        _ = result["items[99]"]


def test_path_access_fill_rate_non_list_with_index() -> None:
    """Test that using index on non-list field in fill rate result raises KeyError."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()

    # Try to use index on non-list field
    with pytest.raises(KeyError, match="Cannot use index on non-list"):
        _ = result["name[0]"]


def test_path_access_field_collection_list_after_non_list() -> None:
    """Test FieldCollection when next segment is list index but current is not list."""

    class Person(BaseModel):
        name: str
        items: list[str]

    person = Person(name="John", items=["a", "b"])

    # Try to access name[0] where name is not a list
    with pytest.raises(KeyError, match="Cannot use index on non-list"):
        _ = person.fields._resolve_path(["name", "[0]"])


def test_path_access_field_collection_list_value_not_list() -> None:
    """Test FieldCollection when Field.value is not a list but index is used."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    # Manually set name to a non-list value but try to use index
    person.fields._fields["name"].value = "string"  # type: ignore

    with pytest.raises(KeyError, match="Cannot use index on non-list"):
        _ = person.fields._resolve_path(["name", "[0]"])


def test_path_access_field_collection_invalid_index_direct() -> None:
    """Test FieldCollection._resolve_path with invalid index directly."""

    class Person(BaseModel):
        items: list[str]

    person = Person(items=["a", "b"])

    # Try to resolve path with invalid index (non-numeric)
    with pytest.raises(KeyError, match="Invalid list index"):
        _ = person.fields._resolve_path(["items", "[abc]"])


def test_path_access_fill_rate_invalid_index_direct() -> None:
    """Test FillRateFieldCollection._resolve_path with invalid index directly."""

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
    result = order.compute_fill_rate()

    # Try to resolve path with invalid index (non-numeric)
    with pytest.raises(KeyError, match="Invalid list index"):
        _ = result.fields._resolve_path(["items", "[abc]"])


def test_path_access_field_collection_list_direct() -> None:
    """Test FieldCollection when current is a list directly."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    # Manually set a field to a list
    from cobjectric.field import Field
    from cobjectric.field_spec import FieldSpec

    person.fields._fields["items"] = Field(
        name="items",
        type=list[str],
        value=["a", "b", "c"],
        spec=FieldSpec(),
    )

    # Access list item directly
    item = person.fields._resolve_path(["items", "[1]"])
    assert item == "b"


def test_path_access_field_collection_list_direct_out_of_range() -> None:
    """Test FieldCollection when current is a list and index is out of range."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    # Manually set a field to a list
    from cobjectric.field import Field
    from cobjectric.field_spec import FieldSpec

    person.fields._fields["items"] = Field(
        name="items",
        type=list[str],
        value=["a"],
        spec=FieldSpec(),
    )

    # Try to access out of range index
    with pytest.raises(KeyError, match="List index.*out of range"):
        _ = person.fields._resolve_path(["items", "[99]"])


def test_path_access_field_collection_list_next_segment_index() -> None:
    """Test FieldCollection when current is list and next segment is index."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    # Manually set a field to a list of BaseModels
    from cobjectric.field import Field
    from cobjectric.field_spec import FieldSpec

    class Item(BaseModel):
        value: str

    person.fields._fields["items"] = Field(
        name="items",
        type=list[Item],
        value=[Item(value="a"), Item(value="b")],
        spec=FieldSpec(),
    )

    # Access list[0].value - should work
    item = person.fields._resolve_path(["items", "[0]"])
    assert isinstance(item, BaseModel)
    assert item.fields.value.value == "a"


def test_path_access_fill_rate_list_direct_field_access() -> None:
    """Test that accessing field directly on list raises KeyError with helpful message."""

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
    result = order.compute_fill_rate()

    # Try to access name directly on items (should use items.name for aggregated or items[0].name for index)
    with pytest.raises(
        KeyError, match="Cannot access.*directly on list field.*Use index"
    ):
        _ = result.fields._resolve_path(["items", "name"])
