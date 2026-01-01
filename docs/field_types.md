# Field Types

Cobjectric supports various field types for defining models. This guide covers all supported field types and their usage.

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

## Supported Types

Only **JSON-compatible types** are supported:
- Primitive types: `str`, `int`, `float`, `bool`
- Lists: `list[T]` where `T` is a supported type
- Dictionaries: `dict` or `dict[K, V]` where `K` and `V` are supported types
- Union types: `T | U` or `t.Union[T, U]` where `T` and `U` are supported types
- Optional types: `T | None` or `t.Optional[T]` where `T` is a supported type
- Nested models: `BaseModel` subclasses

## Unsupported Types

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
import typing as t

# These will all raise UnsupportedTypeError:
class Person(BaseModel):
    data: t.Any      # Not supported
    data: object     # Not supported
    tags: set[str]   # Not supported
    pair: tuple[str, int]  # Not supported
```

## Related Topics

- [BaseModel](base_model.md) - Learn about the base model class
- [Nested Models](nested_models.md) - Learn about nested model structures
- [Field Specifications](field_specs.md) - Learn about Spec() and field normalizers

