# Path Access

You can access fields and nested fields using path notation with `["path.to.field"]`:

## Simple Field Access

```python
class Person(BaseModel):
    name: str
    age: int

person = Person(name="John", age=30)
result = person.compute_fill_rate()

# Access by path
print(result["name"].value)  # 1.0
print(result["age"].value)   # 1.0
```

## Nested Model Access

```python
class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

person = Person.from_dict({
    "name": "John",
    "address": {"street": "123 Main St", "city": "Anytown"},
})
result = person.compute_fill_rate()

# Access nested fields
print(result["address.city"].value)    # 1.0
print(result["address.street"].value) # 1.0
```

## Path Access on BaseModel

You can also use path access directly on `BaseModel` instances:

```python
class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

person = Person.from_dict({
    "name": "John",
    "address": {"street": "123 Main St", "city": "Anytown"},
})

# Access field values by path
print(person["name"].value)           # "John"
print(person["address.city"].value)   # "Anytown"
print(person["address.street"].value) # "123 Main St"
```

## List Index Access

Path access supports list indices with the syntax `[0]`, `[1]`, etc. You can access list elements and their nested fields:

```python
class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    items: list[Item]

order = Order.from_dict({
    "items": [
        {"name": "Apple", "price": 1.0},
        {"name": "Banana", "price": 0.5},
    ],
})

# Access list item by index
item0 = order["items[0]"]  # Returns the Item BaseModel instance
print(item0.fields.name.value)  # "Apple"

# Access nested field through list index
name_field = order["items[0].name"]  # Returns Field
print(name_field.value)  # "Apple"

price_field = order["items[1].price"]
print(price_field.value)  # 0.5

# Fill rate results also support list index access
result = order.compute_fill_rate()
item0_result = result["items[0]"]  # ModelResult
print(item0_result.fields.name.value)  # 1.0

name_result = result["items[0].name"]  # FieldResult
print(name_result.value)  # 1.0

# Multiple nested indices work too
class Address(BaseModel):
    city: str

class Item(BaseModel):
    name: str
    address: Address

class Order(BaseModel):
    items: list[Item]

order = Order.from_dict({
    "items": [
        {"name": "Item1", "address": {"city": "NYC"}},
    ],
})

city_field = order["items[0].address.city"]
print(city_field.value)  # "NYC"

result = order.compute_fill_rate()
city_result = result["items[0].address.city"]
print(city_result.value)  # 1.0
```

**Error Handling**:

```python
# Out of range index
try:
    _ = order["items[99]"]  # Raises KeyError: "List index 99 out of range"
except KeyError:
    pass

# Multiple list indices are also parsed:
try:
    _ = result["orders[0].items[1].name"]  # Also raises KeyError
except KeyError:
    pass
```

**Note**: List index access syntax (`[0]`, `[1]`, etc.) is recognized and parsed, but accessing list elements through path notation currently raises `KeyError` with the message "List index access not yet supported". Full support for list indices in path access will be available in a future version.

## Invalid Paths

Accessing an invalid path raises `KeyError`:

```python
class Person(BaseModel):
    name: str

person = Person(name="John")
result = person.compute_fill_rate()

try:
    _ = result["non_existent"]
except KeyError:
    print("Field not found")

try:
    _ = result["name.invalid"]
except KeyError:
    print("Invalid nested path")

# List index access currently raises KeyError
try:
    _ = result["items[0]"]
except KeyError:
    print("List index access not yet supported")
```

## Related Topics

- [BaseModel](base_model.md) - Learn about the base model class
- [Nested Models](nested_models.md) - Learn about nested model structures
- [Fill Rate](fill_rate.md) - Learn about fill rate computation

## API Reference

See the [API Reference](reference.md#field-results) for result classes that support path access via `__getitem__`.

