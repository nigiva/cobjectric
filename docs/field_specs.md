# Field Specifications and Normalizers

This guide explains how to use field specifications (`Spec()`) and field normalizers to customize field behavior.

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

from cobjectric import MissingValue

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

## Related Topics

- [BaseModel](base_model.md) - Learn about the base model class
- [Fill Rate](fill_rate.md) - Learn about fill rate functions that can be defined in Spec()
- [Similarity](similarity.md) - Learn about similarity functions that can be defined in Spec()

