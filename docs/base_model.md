# BaseModel and Fields

This guide explains how to use `BaseModel`, `Field`, `FieldCollection`, and related concepts.

## Table of Contents

1. [BaseModel](#basemodel)
2. [Fields](#fields)
3. [FieldCollection](#fieldcollection)
4. [MissingValue](#missingvalue)
5. [API Reference](#api-reference)

## BaseModel

`BaseModel` is the base class for defining models with typed fields.

### Defining a Model

Define a model by subclassing `BaseModel` and adding class attributes with type annotations:

```python
from cobjectric import BaseModel

class User(BaseModel):
    username: str
    email: str
    age: int
    is_admin: bool
```

### Creating Instances

Create instances by passing field values as keyword arguments:

```python
user = User(
    username="john_doe",
    email="john@example.com",
    age=30,
    is_admin=False,
)
```

### Model Properties

#### Immutability

Models are immutable after creation. Attempting to set attributes will raise an `AttributeError`:

```python
user.username = "new_name"  # Raises AttributeError
```

#### Missing Fields

Fields that are not provided during instantiation will have the value `MissingValue`:

```python
user = User(username="jane_doe", email="jane@example.com")
print(user.fields.age.value)  # MissingValue
```

#### Type Validation

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
- **specs**: Field specifications (currently not implemented)

### Accessing Fields

Access fields through the `.fields` attribute of a model instance:

```python
user = User(username="bob", email="bob@example.com", age=25)
name_field = user.fields.username
print(name_field.name)   # "username"
print(name_field.type)   # <class 'str'>
print(name_field.value)  # "bob"
```

## FieldCollection

`FieldCollection` is a collection of `Field` instances that provides convenient access to all fields in a model.

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

- **`__getattr__(name: str) -> Field`**: Get a field by name
- **`__iter__() -> Iterator[Field]`**: Iterate over all fields

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

## Examples

### Example 1: Basic Model

```python
from cobjectric import BaseModel

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

product = Product(name="Laptop", price=999.99, in_stock=True)
print(product.fields.name.value)    # "Laptop"
print(product.fields.price.value)   # 999.99
```

### Example 2: Handling Missing Fields

```python
from cobjectric import MissingValue

user = User(username="dave", email="dave@example.com")
# age is not provided
if user.fields.age.value is MissingValue:
    print("User age not provided")
```

### Example 3: Type Validation

```python
# Wrong type for age
user = User(
    username="eve",
    email="eve@example.com",
    age="twenty",
)
print(user.fields.age.value is MissingValue)  # True
```

