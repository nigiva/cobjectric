# BaseModel and Fields

This guide explains how to use `BaseModel`, `Field`, `FieldCollection`, and related concepts.

## Table of Contents

1. [BaseModel](#basemodel)
2. [Nested Models](#nested-models)
3. [List Fields](#list-fields)
4. [Fields](#fields)
5. [FieldCollection](#fieldcollection)
6. [MissingValue](#missingvalue)
7. [API Reference](#api-reference)

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

### Creating Instances from Dictionaries

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

## Nested Models

You can define nested models by using another `BaseModel` subclass as a field type. When creating instances from dictionaries, nested dictionaries are automatically converted to model instances.

### Defining Nested Models

```python
from cobjectric import BaseModel

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
```

### Creating Nested Models from Dictionaries

When using `from_dict`, nested dictionaries are automatically converted to model instances:

```python
person = Person.from_dict({
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
})
```

### Accessing Nested Model Fields

Nested models are accessed directly through the `.fields` attribute, not as `Field` instances:

```python
# Access the nested model
address = person.fields.address  # Returns an Address instance

# Access fields of the nested model
print(address.fields.street.value)  # "123 Main St"
print(address.fields.city.value)   # "Anytown"
```

### Creating Nested Models with Instances

You can also pass model instances directly when creating a model:

```python
address = Address(
    street="123 Main St",
    city="Anytown",
    state="CA",
    zip_code="12345",
    country="USA",
)

person = Person(
    name="John Doe",
    age=30,
    email="john.doe@example.com",
    is_active=True,
    address=address,
)
```

### Deeply Nested Models

You can nest models at multiple levels:

```python
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

person = Person.from_dict({
    "name": "John Doe",
    "address": {
        "street": "123 Main St",
        "city": "Anytown",
        "country": {
            "name": "United States",
            "code": "US",
        },
    },
})

# Access deeply nested fields
print(person.fields.address.fields.country.fields.name.value)  # "United States"
```

### Missing Nested Models

If a nested model is not provided or has an invalid type, it will have the value `MissingValue`:

```python
person = Person.from_dict({
    "name": "John Doe",
    # address is missing
})

print(person.fields.address.value is MissingValue)  # True
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

## List Fields

You can define fields that contain lists of values. Lists can contain primitive types (str, int, float, bool) or nested `BaseModel` instances.

### Defining List Fields

```python
from cobjectric import BaseModel

class Person(BaseModel):
    name: str
    skills: list[str]
    scores: list[int]
```

### List of Primitive Types

```python
person = Person(
    name="John Doe",
    skills=["Python", "JavaScript", "Rust"],
    scores=[85, 90, 95],
)
print(person.fields.skills.value)  # ["Python", "JavaScript", "Rust"]
print(person.fields.scores.value)  # [85, 90, 95]
```

### List of BaseModel Instances

```python
class Experience(BaseModel):
    title: str
    company: str
    start_date: str
    end_date: str

class Person(BaseModel):
    name: str
    experiences: list[Experience]

person = Person.from_dict({
    "name": "John Doe",
    "experiences": [
        {
            "title": "Software Engineer",
            "company": "Tech Corp",
            "start_date": "2020-01-01",
            "end_date": "2022-01-01",
        },
        {
            "title": "Senior Engineer",
            "company": "Big Tech",
            "start_date": "2022-01-01",
            "end_date": "2024-01-01",
        },
    ],
})

print(person.fields.experiences.value)  # [Experience(...), Experience(...)]
print(person.fields.experiences.value[0].fields.title.value)  # "Software Engineer"
```

### List Validation and Partial Filtering

Lists are validated element by element. **Valid elements are kept, invalid elements are filtered out**:

```python
person = Person(
    name="John Doe",
    scores=[85, 90, "invalid", 95],  # Contains a string instead of int
)
print(person.fields.scores.value)  # [85, 90, 95] - invalid element filtered out
```

If **all elements are invalid**, the list will have the value `MissingValue`:

```python
person = Person(
    name="John Doe",
    scores=["a", "b", "c"],  # All elements are invalid
)
print(person.fields.scores.value is MissingValue)  # True
```

For lists of `BaseModel` instances, non-dict elements are filtered out:

```python
person = Person.from_dict({
    "name": "John Doe",
    "experiences": [
        {"title": "Engineer", "company": "Tech Corp"},
        "invalid string",  # This will be filtered out
        {"title": "Senior", "company": "Big Tech"},
    ],
})
print(len(person.fields.experiences.value))  # 2 - only dicts are kept
```

### Empty Lists

Empty lists are valid:

```python
person = Person(name="John Doe", skills=[])
print(person.fields.skills.value)  # []
```

### Missing Lists

Lists that are not provided will have the value `MissingValue`:

```python
person = Person(name="John Doe")
print(person.fields.skills.value is MissingValue)  # True
```

## Optional Fields

You can define optional fields using the `| None` syntax or `t.Optional[T]`:

```python
from cobjectric import BaseModel
import typing as t

class Person(BaseModel):
    name: str
    email: str | None
    phone: t.Optional[str]
```

### Optional Fields with Values

```python
person = Person(name="John Doe", email="john@example.com")
print(person.fields.name.value)  # "John Doe"
print(person.fields.email.value)  # "john@example.com"
```

### Optional Fields with None

```python
person = Person(name="John Doe", email=None)
print(person.fields.email.value)  # None
```

### Missing Optional Fields

If an optional field is not provided, it will have `MissingValue`:

```python
person = Person(name="John Doe")
print(person.fields.email.value is MissingValue)  # True
```

## Union Types

You can define fields that accept multiple types using union syntax:

```python
from cobjectric import BaseModel

class Person(BaseModel):
    name: str
    id: str | int
    status: bool | int
```

### Union Type Validation

The first matching type in the union is used:

```python
person1 = Person(name="John", id="abc123")
print(person1.fields.id.value)  # "abc123"

person2 = Person(name="John", id=123)
print(person2.fields.id.value)  # 123
```

### Union with None

You can combine union types with `None`:

```python
class Person(BaseModel):
    name: str
    metadata: dict[str, int] | None
    scores: list[int] | None

person1 = Person(name="John", metadata={"age": 30})
print(person1.fields.metadata.value)  # {"age": 30}

person2 = Person(name="John", metadata=None)
print(person2.fields.metadata.value)  # None
```

### Union with Complex Types

Unions work with any supported types:

```python
class Person(BaseModel):
    name: str
    data: list[int] | dict[str, int]

person1 = Person(name="John", data=[1, 2, 3])
print(person1.fields.data.value)  # [1, 2, 3]

person2 = Person(name="John", data={"a": 1, "b": 2})
print(person2.fields.data.value)  # {"a": 1, "b": 2}
```

## Typed Dict Fields

You can define fields with typed dictionaries using `dict[K, V]` syntax:

```python
from cobjectric import BaseModel

class Person(BaseModel):
    name: str
    scores: dict[str, int]
    metadata: dict[str, str]
```

### Typed Dict Validation

Typed dicts validate both keys and values:

```python
person = Person(
    name="John",
    scores={"math": 90, "english": 85, "science": 95}
)
print(person.fields.scores.value)  # {"math": 90, "english": 85, "science": 95}
```

### Typed Dict Partial Filtering

Invalid entries are automatically filtered out:

```python
person = Person(
    name="John",
    scores={"math": 90, "english": "invalid", "science": 95}
)
print(person.fields.scores.value)  # {"math": 90, "science": 95}
```

If all entries are invalid, the field will have `MissingValue`:

```python
person = Person(
    name="John",
    scores={"math": "invalid", "english": "bad"}
)
print(person.fields.scores.value is MissingValue)  # True
```

### Empty Typed Dict

Empty dictionaries are valid:

```python
person = Person(name="John", scores={})
print(person.fields.scores.value)  # {}
```

### Nested Typed Dicts

Typed dicts can be nested recursively:

```python
class Person(BaseModel):
    name: str
    nested_scores: dict[str, dict[str, int]]

person = Person(
    name="John",
    nested_scores={
        "semester1": {"math": 90, "english": 85},
        "semester2": {"math": 95, "english": 80},
    }
)
print(person.fields.nested_scores.value)
# {"semester1": {"math": 90, "english": 85}, "semester2": {"math": 95, "english": 80}}
```

### Typed Dict with List Values

Typed dicts can have list values:

```python
class Person(BaseModel):
    name: str
    scores_by_subject: dict[str, list[int]]

person = Person(
    name="John",
    scores_by_subject={
        "math": [90, 85, 95],
        "english": [80, 85],
    }
)
print(person.fields.scores_by_subject.value)
# {"math": [90, 85, 95], "english": [80, 85]}
```

### Typed Dict with Union Values

Typed dicts can have union types as values:

```python
class Person(BaseModel):
    name: str
    scores: dict[str, int | None]

person = Person(
    name="John",
    scores={"math": 90, "english": None, "science": 85}
)
print(person.fields.scores.value)  # {"math": 90, "english": None, "science": 85}
```

### Bare Dict Type

You can also use bare `dict` type (without type arguments) for untyped dictionaries:

```python
class Person(BaseModel):
    name: str
    metadata: dict

person = Person(name="John", metadata={"key": "value", "num": 42})
print(person.fields.metadata.value)  # {"key": "value", "num": 42}
```

**Note**: Bare `dict` accepts any dictionary without validation. For type safety, prefer `dict[K, V]`.

### Supported Types

Only **JSON-compatible types** are supported:
- Primitive types: `str`, `int`, `float`, `bool`
- Lists: `list[T]` where `T` is a supported type
- Dictionaries: `dict` or `dict[K, V]` where `K` and `V` are supported types
- Union types: `T | U` or `t.Union[T, U]` where `T` and `U` are supported types
- Optional types: `T | None` or `t.Optional[T]` where `T` is a supported type
- Nested models: `BaseModel` subclasses

### Unsupported Types

The following types are **not supported** and will raise exceptions:

**Union types in lists:**
```python
from cobjectric import UnsupportedListTypeError

class Person(BaseModel):
    name: str
    mixed: list[str | int]  # This will raise an error

try:
    person = Person(name="John Doe", mixed=["a", 1])
except UnsupportedListTypeError as e:
    print(f"Error: {e}")
    # Output: Error: Unsupported list type: list[str | int]. List fields must contain
    # a single type (e.g., list[str], list[int], list[MyModel]). Union types like
    # list[str | int] are not supported.
```

**List without type arguments:**
```python
from cobjectric import MissingListTypeArgError

class Person(BaseModel):
    name: str
    items: list  # This will raise an error

try:
    person = Person(name="John Doe", items=["a", 1])
except MissingListTypeArgError as e:
    print(f"Error: {e}")
    # Output: Error: List type must specify an element type. Use list[str], list[int],
    # list[MyModel], etc. instead of bare 'list'.
```

**Other unsupported types:**
```python
from cobjectric import UnsupportedTypeError

# These will all raise UnsupportedTypeError:
class Person(BaseModel):
    data: t.Any      # Not supported
    data: object     # Not supported
    tags: set[str]   # Not supported
    pair: tuple[str, int]  # Not supported
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

## Field Specifications (Spec)

You can add metadata to fields using the `Spec()` function. This is useful for documentation, validation hints, or any other metadata you want to associate with a field.

### Using Spec

Define fields with specifications using the `Spec()` function:

```python
from cobjectric import BaseModel, Spec

class Person(BaseModel):
    name: str = Spec(metadata={"description": "The name of the person"})
    age: int = Spec()
    email: str
    is_active: bool
```

### Accessing Field Specifications

Access the specification through the field's `.spec` attribute:

```python
person = Person.from_dict({
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
    "is_active": True,
})

# Access spec with metadata
print(person.fields.name.spec.metadata)
# {"description": "The name of the person"}

# Fields without Spec() have a default FieldSpec with empty metadata
print(person.fields.email.spec.metadata)
# {}
```

### Default FieldSpec

Fields that don't explicitly use `Spec()` automatically get a default `FieldSpec` with empty metadata:

```python
class Person(BaseModel):
    name: str  # No Spec() used

person = Person.from_dict({"name": "John Doe"})
assert person.fields.name.spec.metadata == {}  # Empty dict by default
```

### FieldSpec Attributes

- **metadata**: A dictionary containing field metadata (dict[str, Any])

### Spec Function

The `Spec()` function creates a `FieldSpec` instance. It accepts an optional `metadata` parameter:

```python
# With metadata
name: str = Spec(metadata={"description": "Name", "required": True})

# Without metadata (empty dict)
age: int = Spec()

# None metadata is treated as empty dict
email: str = Spec(metadata=None)  # Same as Spec()
```

**Note**: The function returns `Any` for type checking purposes, allowing it to be used in type annotations without causing type errors. This follows the same pattern as Pydantic's `Field()` function.

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

### Example 4: Creating from Dictionary

```python
from cobjectric import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str
    is_active: bool

# Create from dictionary
person = Person.from_dict({
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
    "is_active": True,
})

print(person.fields.name.value)  # "John Doe"
print(person.fields.age.value)   # 30
```

### Example 5: Nested Models

```python
from cobjectric import BaseModel

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

# Create from dictionary with nested model
person = Person.from_dict({
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
})

print(person.fields.name.value)  # "John Doe"
print(person.fields.address.fields.street.value)  # "123 Main St"
print(person.fields.address.fields.city.value)    # "Anytown"
```

