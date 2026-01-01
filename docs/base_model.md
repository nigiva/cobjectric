# BaseModel

`BaseModel` is the base class for defining models with typed fields.

## Defining a Model

Define a model by subclassing `BaseModel` and adding class attributes with type annotations:

```python
from cobjectric import BaseModel

class User(BaseModel):
    username: str
    email: str
    age: int
    is_admin: bool
```

## Creating Instances

Create instances by passing field values as keyword arguments:

```python
user = User(
    username="john_doe",
    email="john@example.com",
    age=30,
    is_admin=False,
)
```

## Creating Instances from Dictionaries

You can also create instances from dictionaries using the `from_dict` class method:

```python
user = User.from_dict({
    "username": "john_doe",
    "email": "john@example.com",
    "age": 30,
    "is_admin": False,
})
```

This is particularly useful when working with JSON data or dictionaries from external sources:

```python
data = {
    "username": "jane_doe",
    "email": "jane@example.com",
    "age": 25,
    "is_admin": True,
}
user = User.from_dict(data)
```

## Model Properties

### Immutability

Models are immutable after creation. Attempting to set attributes will raise an `AttributeError`:

```python
user.username = "new_name"  # Raises AttributeError
```

### Missing Fields

Fields that are not provided during instantiation will have the value `MissingValue`:

```python
user = User(username="jane_doe", email="jane@example.com")
print(user.fields.age.value)  # MissingValue
```

### Type Validation

Fields with values of incorrect types will also have the value `MissingValue`:

```python
user = User(
    username="alice",
    email="alice@example.com",
    age="thirty",  # Wrong type
)
print(user.fields.age.value)  # MissingValue
```

## Fields

A `Field` represents a single attribute in a model. It contains metadata about the field including its name, type, value, and specifications.

### Field Attributes

- **name**: The name of the field (str)
- **type**: The type annotation of the field (type)
- **value**: The current value of the field (any type, or MissingValue)
- **spec**: The field specification (FieldSpec)

### Accessing Fields

Access fields through the `.fields` attribute of a model instance:

```python
user = User(username="bob", email="bob@example.com", age=25)
name_field = user.fields.username
print(name_field.name)   # "username"
print(name_field.type)   # <class 'str'>
print(name_field.value)  # "bob"
print(name_field.spec)   # FieldSpec(metadata={})
```

## FieldCollection

`FieldCollection` is a collection of `Field` instances or `BaseModel` instances (for nested models) that provides convenient access to all fields in a model.

### Accessing Fields by Attribute

```python
user = User(
    username="alice",
    email="alice@example.com",
    age=28,
    is_admin=True,
)
print(user.fields.username.value)  # "alice"
print(user.fields.is_admin.value)  # True
```

### Iterating Over Fields

```python
for field in user.fields:
    print(f"{field.name}: {field.value}")
# Output:
# username: alice
# email: alice@example.com
# age: 28
# is_admin: True
```

### Field Collection Representation

```python
print(user.fields)
# FieldCollection(
#   username=Field(...),
#   email=Field(...),
#   age=Field(...),
#   is_admin=Field(...)
# )
```

## MissingValue

`MissingValue` is a sentinel value that indicates a field is missing or has an invalid type.

### Using MissingValue

```python
from cobjectric import MissingValue

user = User(username="charlie")
print(user.fields.email.value is MissingValue)  # True
```

### Checking for Missing Fields

```python
if user.fields.age.value is MissingValue:
    print("Age field is missing")
```

## API Reference

### BaseModel

#### Methods and Properties

- **`fields`** (property): Returns a `FieldCollection` containing all fields
- **`__init__(**kwargs)`**: Initialize the model with field values
- **`from_dict(data: dict[str, Any])`** (classmethod): Create a model instance from a dictionary
- **`__repr__()`**: Returns a string representation of the model (similar to Pydantic)

#### String Representation

BaseModel instances have a `__repr__` method that provides a readable string representation:

```python
class Address(BaseModel):
    city: str
    zip: str

class Person(BaseModel):
    name: str
    age: int
    address: Address

person = Person.from_dict({
    "name": "John",
    "age": 30,
    "address": {"city": "NYC", "zip": "10001"},
})

print(repr(person))
# Output: Person(name='John', age=30, address=Address(city='NYC', zip='10001'))
```

Missing fields are displayed as `MISSING`:

```python
person = Person(name="John")
print(repr(person))
# Output: Person(name='John', age=MISSING, address=MISSING)
```

Lists are properly represented:

```python
class Item(BaseModel):
    name: str

class Order(BaseModel):
    items: list[Item]

order = Order.from_dict({
    "items": [
        {"name": "Apple"},
        {"name": "Banana"},
    ],
})
print(repr(order))
# Output: Order(items=[Item(name='Apple'), Item(name='Banana')])
```

### Field

#### Constructor

```python
Field(name: str, type: type, value: Any, specs: Any)
```

#### Attributes

- **`name`**: Field name (str)
- **`type`**: Field type (type)
- **`value`**: Field value (Any or MissingValue)
- **`specs`**: Field specifications (Any)

### FieldCollection

#### Methods

- **`__getattr__(name: str) -> Field | BaseModel`**: Get a field by name (returns a `Field` for primitive types or a `BaseModel` instance for nested models)
- **`__iter__() -> Iterator[Field | BaseModel]`**: Iterate over all fields

### Exceptions

#### CobjectricError

Base exception class for all Cobjectric errors.

```python
from cobjectric import CobjectricError

try:
    # Your code here
    pass
except CobjectricError as e:
    print(f"Error: {e}")
```

## Related Topics

- [Field Types](field_types.md) - Learn about different field types (Optional, Union, Dict, List)
- [Nested Models](nested_models.md) - Learn about nested model structures
- [Field Specifications](field_specs.md) - Learn about Spec() and field normalizers
- [Path Access](path_access.md) - Learn about accessing fields by path notation
