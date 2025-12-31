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

## Field Normalizers

You can define normalizers to transform field values before type validation. Normalizers are applied in a specific order: first the normalizer from `Spec(normalizer=...)`, then any `@field_normalizer` decorators in declaration order.

### Using Spec(normalizer=...)

You can define a normalizer directly in the `Spec()` function:

```python
from cobjectric import BaseModel, Spec

class Person(BaseModel):
    name: str = Spec(normalizer=lambda x: x.lower())
    age: int = Spec(normalizer=lambda x: int(x))

person = Person(name="JOHN DOE", age="30")
print(person.fields.name.value)  # "john doe"
print(person.fields.age.value)   # 30
```

### Using @field_normalizer Decorator

You can also define normalizers using the `@field_normalizer` decorator:

```python
from cobjectric import BaseModel, field_normalizer
import typing as t

class Person(BaseModel):
    name: str
    email: str

    @field_normalizer("name", "email")
    def normalize_strings(x: t.Any) -> str:
        return str(x).strip().lower()

person = Person(name="  JOHN DOE  ", email="  JOHN@EXAMPLE.COM  ")
print(person.fields.name.value)   # "john doe"
print(person.fields.email.value)  # "john@example.com"
```

### Pattern Matching

The `@field_normalizer` decorator supports glob patterns to match multiple fields:

```python
class Person(BaseModel):
    name_first: str
    name_last: str
    age: int

    @field_normalizer("name_*")
    def normalize_name_fields(x: t.Any) -> str:
        return str(x).strip().title()

person = Person(name_first="  john  ", name_last="  DOE  ", age=30)
print(person.fields.name_first.value)  # "John"
print(person.fields.name_last.value)   # "Doe"
print(person.fields.age.value)        # 30
```

### Combining Multiple Normalizers

You can combine `Spec(normalizer=...)` and `@field_normalizer` decorators. The Spec normalizer runs first, followed by decorator normalizers in declaration order:

```python
class Person(BaseModel):
    name: str = Spec(normalizer=lambda x: x.lower())

    @field_normalizer("name")
    def trim(x: t.Any) -> str:
        return str(x).strip()

    @field_normalizer("name")
    def capitalize_first(x: t.Any) -> str:
        return str(x).capitalize()

person = Person(name="  JOHN DOE  ")
# Order: lowercase -> trim -> capitalize_first
print(person.fields.name.value)  # "John doe"
```

### Accessing the Combined Normalizer

The combined normalizer (from Spec + decorators) is stored in `spec.normalizer`:

```python
class Person(BaseModel):
    name: str = Spec(normalizer=lambda x: x.lower())

    @field_normalizer("name")
    def trim(x: t.Any) -> str:
        return str(x).strip()

person = Person(name="  TEST  ")
# The combined normalizer is available
normalizer = person.fields.name.spec.normalizer
assert normalizer is not None
result = normalizer("  TEST  ")
print(result)  # "test"
```

### Error Handling

Normalizer exceptions propagate naturally. If you want to handle errors gracefully, wrap your normalizer function with try/except:

```python
def safe_int_normalizer(x: t.Any) -> int | None:
    try:
        return int(x)
    except (ValueError, TypeError):
        return None  # Will become MissingValue due to type mismatch
        # You can also return MissingValue directly

class Person(BaseModel):
    age: int = Spec(normalizer=safe_int_normalizer)

person = Person(age="invalid")
print(person.fields.age.value is MissingValue)  # True
```

### Normalizer Behavior

- **Normalizers are applied before type validation**: The normalized value is then validated against the field type.
- **If normalizer returns incompatible type**: The field will have `MissingValue` after type validation.
- **If value is MissingValue**: Normalizers are not applied (the value remains `MissingValue`).
- **Normalizers work with all field types**: Including primitives, lists, dicts, optional fields, and nested models.

### Examples

**Normalizer on optional field:**

```python
class Person(BaseModel):
    email: str | None = Spec(normalizer=lambda x: x.lower() if x else None)

person1 = Person(email="JOHN@EXAMPLE.COM")
print(person1.fields.email.value)  # "john@example.com"

person2 = Person(email=None)
print(person2.fields.email.value)  # None
```

**Normalizer on list field:**

```python
class Person(BaseModel):
    tags: list[str] = Spec(normalizer=lambda x: [t.lower() for t in x])

person = Person(tags=["TAG1", "TAG2", "TAG3"])
print(person.fields.tags.value)  # ["tag1", "tag2", "tag3"]
```

**Normalizer on dict field:**

```python
class Person(BaseModel):
    metadata: dict = Spec(
        normalizer=lambda x: {k.lower(): v for k, v in x.items()}
    )

person = Person(metadata={"KEY1": "value1", "KEY2": "value2"})
print(person.fields.metadata.value)  # {"key1": "value1", "key2": "value2"}
```

## Fill Rate

Fill rate is a metric that measures how "complete" or "filled" a field value is. It's a float between 0.0 and 1.0, where:
- `0.0` means the field is completely empty or missing
- `1.0` means the field is completely filled
- Values in between represent partial completeness

Fill rate is particularly useful for data quality assessment, where you want to measure how complete your data is across multiple fields.

### Using Spec(fill_rate_func=...)

You can define a custom fill rate function directly in the `Spec()` function:

```python
from cobjectric import BaseModel, Spec

class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: len(x) / 100)
    age: int
    email: str

person = Person(name="John Doe", age=30, email="john@example.com")
result = person.compute_fill_rate()

print(result.fields.name.value)  # 0.08 (len("John Doe") = 8, 8/100 = 0.08)
print(result.fields.age.value)   # 1.0 (default: present = 1.0)
print(result.fields.email.value) # 1.0 (default: present = 1.0)
```

### Using @fill_rate_func Decorator

You can also define fill rate functions using the `@fill_rate_func` decorator:

```python
from cobjectric import BaseModel, fill_rate_func
import typing as t

class Person(BaseModel):
    name: str
    email: str
    age: int

    @fill_rate_func("name", "email")
    def fill_rate_name_email(x: t.Any) -> float:
        return len(x) / 100 if x is not MissingValue else 0.0

    @fill_rate_func("age")
    def fill_rate_age(x: t.Any) -> float:
        return 1.0 if x is not MissingValue else 0.0

person = Person(name="John", email="john@example.com", age=30)
result = person.compute_fill_rate()

print(result.fields.name.value)  # 0.04 (len("John") = 4, 4/100 = 0.04)
print(result.fields.email.value) # 0.16 (len("john@example.com") = 16, 16/100 = 0.16)
print(result.fields.age.value)   # 1.0
```

### Pattern Matching

The `@fill_rate_func` decorator supports glob patterns to match multiple fields:

```python
class Person(BaseModel):
    name_first: str
    name_last: str
    age: int

    @fill_rate_func("name_*")
    def fill_rate_name_fields(x: t.Any) -> float:
        return len(x) / 100 if x is not MissingValue else 0.0

person = Person(name_first="John", name_last="Doe", age=30)
result = person.compute_fill_rate()

print(result.fields.name_first.value)  # 0.04
print(result.fields.name_last.value)   # 0.03
print(result.fields.age.value)        # 1.0
```

### Default Fill Rate Function

If no `fill_rate_func` is specified, the default behavior is:
- Returns `0.0` if the field value is `MissingValue`
- Returns `1.0` otherwise

### Fill Rate Validation

Fill rate functions must return a float (or int convertible to float) between 0.0 and 1.0. If a function returns an invalid value, `InvalidFillRateValueError` is raised:

```python
from cobjectric import InvalidFillRateValueError

class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: 1.5)  # Invalid: > 1.0

person = Person(name="John")
try:
    result = person.compute_fill_rate()
except InvalidFillRateValueError as e:
    print(f"Error: {e}")
```

### Duplicate Fill Rate Functions

A field can only have one `fill_rate_func`. If multiple functions are defined (via `Spec()` and `@fill_rate_func`, or multiple decorators), `DuplicateFillRateFuncError` is raised:

```python
from cobjectric import DuplicateFillRateFuncError

class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: 0.5)

    @fill_rate_func("name")
    def fill_rate_name(x: t.Any) -> float:
        return 0.6

person = Person(name="John")
try:
    result = person.compute_fill_rate()
except DuplicateFillRateFuncError as e:
    print(f"Error: {e}")
```

### Fill Rate on Normalized Values

Fill rate functions are applied to the **normalized value** (after normalizers have been applied):

```python
class Person(BaseModel):
    name: str = Spec(
        normalizer=lambda x: x.lower(),
        fill_rate_func=lambda x: len(x) / 100,
    )

person = Person(name="JOHN DOE")
result = person.compute_fill_rate()

# Normalized value is "john doe" (len=8), not "JOHN DOE" (len=8)
print(result.fields.name.value)  # 0.08
```

### Computing Fill Rate

To compute fill rate for all fields in a model, call the `compute_fill_rate()` method:

```python
class Person(BaseModel):
    name: str
    age: int
    email: str

person = Person(name="John Doe", age=30)
result = person.compute_fill_rate()

print(result.fields.name.value)  # 1.0
print(result.fields.age.value)   # 1.0
print(result.fields.email.value) # 0.0 (missing)
```

### FillRateModelResult

The `compute_fill_rate()` method returns a `FillRateModelResult` object that provides:

- **Field access**: `result.fields.name` returns a `FillRateFieldResult` with the fill rate value and weight
- **Statistical methods**: `mean()` (weighted), `max()`, `min()`, `std()`, `var()`, `quantile(q)`

```python
class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: 0.3)
    age: int = Spec(fill_rate_func=lambda x: 0.8)
    email: str = Spec(fill_rate_func=lambda x: 0.5)

person = Person(name="John", age=30, email="john@example.com")
result = person.compute_fill_rate()

print(result.mean())      # 0.533... (weighted average of 0.3, 0.8, 0.5)
print(result.max())       # 0.8
print(result.min())       # 0.3
print(result.std())       # Standard deviation
print(result.var())       # Variance
print(result.quantile(0.25))  # 25th percentile
print(result.quantile(0.50))  # 50th percentile (median)
print(result.quantile(0.75))  # 75th percentile
```

**Note**: The `mean()` method calculates a **weighted mean** if weights are specified (see [Weighted Mean](#weighted-mean) section below). By default, all fields have weight `1.0`, so the mean is equivalent to a simple average.

### Weighted Mean

By default, all fields have a weight of `1.0`, which means the mean is calculated as a simple average. However, you can assign different weights to fields to calculate a **weighted mean**. This is useful when some fields are more important than others in your data quality assessment.

#### Using Weight in Spec()

You can set a weight directly in `Spec()`:

```python
class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: 0.5, weight=2.0)
    age: int = Spec(fill_rate_func=lambda x: 1.0, weight=1.0)

person = Person(name="John", age=30)
result = person.compute_fill_rate()

# Weighted mean: (0.5 * 2.0 + 1.0 * 1.0) / (2.0 + 1.0) = 2.0 / 3.0 = 0.666...
print(result.mean())  # 0.666...
```

#### Using Weight in Decorator

You can also set a weight in the `@fill_rate_func` decorator:

```python
class Person(BaseModel):
    name: str
    age: int

    @fill_rate_func("name", weight=2.0)
    def fill_rate_name(x: t.Any) -> float:
        return 0.5 if x is not MissingValue else 0.0

    @fill_rate_func("age", weight=1.0)
    def fill_rate_age(x: t.Any) -> float:
        return 1.0 if x is not MissingValue else 0.0

person = Person(name="John", age=30)
result = person.compute_fill_rate()

print(result.fields.name.weight)  # 2.0
print(result.fields.age.weight)   # 1.0
print(result.mean())              # 0.666... (weighted mean)
```

#### Decorator Weight Overrides Spec Weight

If both `Spec(fill_rate_weight=...)` and `@fill_rate_func(..., weight=...)` are defined for the same field, the decorator weight takes precedence:

```python
class Person(BaseModel):
    name: str = Spec(fill_rate_weight=1.0)

    @fill_rate_func("name", weight=2.0)
    def fill_rate_name(x: t.Any) -> float:
        return 1.0 if x is not MissingValue else 0.0

person = Person(name="John")
result = person.compute_fill_rate()

# Decorator weight (2.0) overrides Spec weight (1.0)
print(result.fields.name.weight)  # 2.0
```

#### Weight Validation

Weight must be `>= 0.0`. Negative weights will raise `InvalidWeightError`:

```python
from cobjectric import InvalidWeightError

# This will raise InvalidWeightError
try:
    Spec(weight=-1.0)
except InvalidWeightError as e:
    print(f"Error: {e}")

# This will also raise InvalidWeightError
try:
    @fill_rate_func("name", weight=-1.0)
    def fill_rate_name(x: t.Any) -> float:
        return 1.0
except InvalidWeightError as e:
    print(f"Error: {e}")
```

#### Weighted Mean Formula

The weighted mean is calculated as:

```
weighted_mean = sum(value * weight) / sum(weight)
```

This means:
- You don't need weights to sum to 1.0
- Fields with higher weights have more influence on the mean
- Fields with weight `0.0` don't contribute to the mean (but are still included in max/min)
- If all weights sum to 0.0, the mean returns `0.0`

#### Weighted Mean with Nested Models

Weights work recursively with nested models:

```python
class Address(BaseModel):
    street: str = Spec(fill_rate_func=lambda x: 0.5, weight=2.0)
    city: str = Spec(fill_rate_func=lambda x: 1.0, weight=1.0)

class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: 0.8, weight=1.0)
    address: Address

person = Person.from_dict({
    "name": "John",
    "address": {"street": "123 Main St", "city": "Anytown"},
})
result = person.compute_fill_rate()

# Weighted mean across all fields (including nested):
# (0.8 * 1.0 + 0.5 * 2.0 + 1.0 * 1.0) / (1.0 + 2.0 + 1.0) = 2.8 / 4.0 = 0.7
print(result.mean())  # 0.7
```

#### Max and Min with Weights

The `max()` and `min()` methods are **not affected by weights** - they simply return the maximum and minimum fill rate values:

```python
class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: 0.3, fill_rate_weight=2.0)
    age: int = Spec(fill_rate_func=lambda x: 0.8, fill_rate_weight=0.5)

person = Person(name="John", age=30)
result = person.compute_fill_rate()

print(result.max())  # 0.8 (not affected by weights)
print(result.min())  # 0.3 (not affected by weights)
print(result.mean())  # 0.466... (weighted mean)
```

### Fill Rate with Nested Models

Fill rate computation works recursively with nested models:

```python
class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

person = Person.from_dict({
    "name": "John Doe",
    "address": {
        "street": "123 Main St",
        "city": "Anytown",
    },
})
result = person.compute_fill_rate()

# Access fill rate for top-level field
print(result.fields.name.value)  # 1.0

# Access fill rate for nested model (returns FillRateModelResult)
nested_result = result.fields.address
print(nested_result.fields.street.value)  # 1.0
print(nested_result.fields.city.value)   # 1.0

# Statistical methods aggregate across all fields (including nested)
print(result.mean())  # 1.0 (all fields are present)
```

### Missing Nested Models

If a nested model is missing (has `MissingValue`), all its fields are considered to have a fill rate of `0.0`:

```python
class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

person = Person(name="John Doe")  # address is missing
result = person.compute_fill_rate()

print(result.fields.name.value)  # 1.0
nested_result = result.fields.address
print(nested_result.fields.street.value)  # 0.0
print(nested_result.fields.city.value)    # 0.0
```

## Fill Rate Accuracy

Fill rate accuracy measures whether fields are correctly filled compared to an expected model, regardless of the actual values. It's different from fill rate, which measures completeness. Fill rate accuracy checks if the "filled" or "missing" state matches between two models.

### Concept

Fill rate accuracy returns:
- `1.0` if both models have the same state (both filled or both missing)
- `0.0` if one is filled and the other is missing

| got | expected | accuracy |
|-----|----------|----------|
| filled | filled | 1.0 |
| MissingValue | MissingValue | 1.0 |
| filled | MissingValue | 0.0 |
| MissingValue | filled | 0.0 |

### Computing Fill Rate Accuracy

To compute fill rate accuracy, call the `compute_fill_rate_accuracy()` method with an expected model:

```python
class Person(BaseModel):
    name: str
    age: int
    email: str

person_got = Person.from_dict({"name": "John", "age": 30})
person_expected = Person.from_dict({"name": "Jane", "email": "jane@example.com"})

result = person_got.compute_fill_rate_accuracy(person_expected)

# name: both filled -> 1.0
print(result.fields.name.value)  # 1.0
# age: got filled, expected missing -> 0.0
print(result.fields.age.value)    # 0.0
# email: got missing, expected filled -> 0.0
print(result.fields.email.value)  # 0.0
```

### Using Spec(fill_rate_accuracy_func=...)

You can define a custom fill rate accuracy function in `Spec()`:

```python
class Person(BaseModel):
    name: str = Spec(
        fill_rate_accuracy_func=lambda got, exp: 0.8
        if (got is not MissingValue) == (exp is not MissingValue)
        else 0.0
    )
    age: int

person_got = Person(name="John", age=30)
person_expected = Person(name="Jane", age=25)

result = person_got.compute_fill_rate_accuracy(person_expected)
print(result.fields.name.value)  # 0.8 (custom function)
print(result.fields.age.value)   # 1.0 (default function)
```

### Using @fill_rate_accuracy_func Decorator

You can also use the `@fill_rate_accuracy_func` decorator:

```python
from cobjectric import BaseModel, fill_rate_accuracy_func
import typing as t

class Person(BaseModel):
    name: str
    email: str

    @fill_rate_accuracy_func("name", "email")
    def accuracy_name_email(got: t.Any, expected: t.Any) -> float:
        return 0.9 if (got is not MissingValue) == (expected is not MissingValue) else 0.0

person_got = Person(name="John", email="john@example.com")
person_expected = Person(name="Jane", email="jane@example.com")

result = person_got.compute_fill_rate_accuracy(person_expected)
print(result.fields.name.value)  # 0.9
print(result.fields.email.value) # 0.9
```

### Separate Weights for Fill Rate Accuracy

Fill rate accuracy uses its own weight system, separate from fill rate weights:

```python
class Person(BaseModel):
    name: str = Spec(
        fill_rate_func=lambda x: 0.5,
        fill_rate_weight=2.0,  # Weight for compute_fill_rate()
        fill_rate_accuracy_func=lambda got, exp: 1.0
        if (got is not MissingValue) == (exp is not MissingValue)
        else 0.0,
        fill_rate_accuracy_weight=1.5,  # Weight for compute_fill_rate_accuracy()
    )

person_got = Person(name="John")
person_expected = Person(name="Jane")

# Fill rate result uses fill_rate_weight
fill_rate_result = person_got.compute_fill_rate()
print(fill_rate_result.fields.name.weight)  # 2.0

# Accuracy result uses fill_rate_accuracy_weight
accuracy_result = person_got.compute_fill_rate_accuracy(person_expected)
print(accuracy_result.fields.name.weight)  # 1.5
```

### Fill Rate Accuracy with Nested Models

Fill rate accuracy works recursively with nested models:

```python
class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

person_got = Person.from_dict({
    "name": "John",
    "address": {"street": "123 Main St", "city": "Anytown"},
})
person_expected = Person.from_dict({
    "name": "Jane",
    "address": {"street": "456 Oak Ave", "city": "Somewhere"},
})

result = person_got.compute_fill_rate_accuracy(person_expected)

# Both have name -> 1.0
print(result.fields.name.value)  # 1.0

# Both have address with all fields -> 1.0 for nested fields
print(result.fields.address.fields.street.value)  # 1.0
print(result.fields.address.fields.city.value)   # 1.0
```

### Duplicate Fill Rate Accuracy Functions

A field can only have one `fill_rate_accuracy_func`. If multiple functions are defined, `DuplicateFillRateAccuracyFuncError` is raised:

```python
from cobjectric import DuplicateFillRateAccuracyFuncError

class Person(BaseModel):
    name: str = Spec(fill_rate_accuracy_func=lambda got, exp: 0.5)

    @fill_rate_accuracy_func("name")
    def accuracy_name(got: t.Any, expected: t.Any) -> float:
        return 0.6

person_got = Person(name="John")
person_expected = Person(name="Jane")

try:
    result = person_got.compute_fill_rate_accuracy(person_expected)
except DuplicateFillRateAccuracyFuncError as e:
    print(f"Error: {e}")
```

## Path Access

You can access fields and nested fields using path notation with `["path.to.field"]`:

### Simple Field Access

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

### Nested Model Access

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

### Path Access on BaseModel

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

### List Index Access (Coming Soon)

Path access supports parsing list indices with the syntax `[0]`, `[1]`, etc. The syntax is recognized and parsed, but full support for accessing list elements is not yet implemented:

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
result = order.compute_fill_rate()

# Syntax is parsed but raises KeyError when used
# This will work in a future version:
try:
    _ = result["items[0].name"]  # Raises KeyError: "List index access not yet supported"
except KeyError:
    pass

# Multiple list indices are also parsed:
try:
    _ = result["orders[0].items[1].name"]  # Also raises KeyError
except KeyError:
    pass
```

**Note**: List index access syntax (`[0]`, `[1]`, etc.) is recognized and parsed, but accessing list elements through path notation currently raises `KeyError` with the message "List index access not yet supported". Full support for list indices in path access will be available in a future version.

### Invalid Paths

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

