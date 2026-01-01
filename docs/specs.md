# Pre-defined Specs

Cobjectric provides pre-defined Specs optimized for common field types. These Specs come with sensible defaults for normalization, similarity functions, and other settings.

## Available Specs

### KeywordSpec

Optimized for identifiers, codes, enums, and other exact-match fields.

**Features:**
- Exact similarity matching
- Optional whitespace stripping (default: enabled)
- Optional int-to-string conversion (default: enabled)
  - Useful when IDs come as integers from JSON but should be compared as strings

**Example:**
```python
from cobjectric import BaseModel
from cobjectric.specs import KeywordSpec

class Person(BaseModel):
    id: str = KeywordSpec()  # Converts 123 → "123"
    code: str = KeywordSpec(strip=True)

person1 = Person(id="  ABC123  ", code="XYZ")
assert person1.fields.id.value == "ABC123"  # Stripped

person2 = Person(id=123)  # int from JSON
assert person2.fields.id.value == "123"  # Converted to string
```

**Parameters:**
- `strip`: Strip leading/trailing whitespace (default: `True`)
- `convert_int_to_str`: Convert int values to string (default: `True`)
  - Useful when IDs come as integers from JSON but should be compared as strings

### TextSpec

Optimized for free-form text fields with fuzzy matching.

**Features:**
- Fuzzy similarity matching (using rapidfuzz)
- Comprehensive text normalization:
  - Lowercase conversion (default: enabled)
  - Whitespace stripping (default: enabled)
  - Space collapsing (default: enabled)
  - Accent removal (default: enabled)

**Example:**
```python
from cobjectric import BaseModel
from cobjectric.specs import TextSpec

class Person(BaseModel):
    name: str = TextSpec()
    description: str = TextSpec(lower=False, remove_accents=False)

person = Person(name="  JOHN DOE  ", description="Café")
assert person.fields.name.value == "john doe"  # Normalized
assert person.fields.description.value == "Café"  # Preserved case and accents
```

**Parameters:**
- `lower`: Convert to lowercase (default: `True`)
- `strip`: Strip leading/trailing whitespace (default: `True`)
- `collapse_spaces`: Collapse multiple spaces to single space (default: `True`)
- `remove_accents`: Remove accents from characters (default: `True`)
- `scorer`: rapidfuzz scorer to use (default: `"WRatio"`)

### NumericSpec

Optimized for numeric fields (int, float) with intelligent type coercion.

**Features:**
- Numeric similarity with optional tolerance
- Automatic type coercion (JSON Number → Python int/float)
- Handles Union types (e.g., `int | None`)

**Example:**
```python
from cobjectric import BaseModel
from cobjectric.specs import NumericSpec

class Person(BaseModel):
    age: int = NumericSpec()  # 30.0 from JSON → 30 (int)
    score: float = NumericSpec(max_difference=0.5)

person = Person(age=30.0, score=85)
assert person.fields.age.value == 30
assert isinstance(person.fields.age.value, int)
assert person.fields.score.value == 85.0
assert isinstance(person.fields.score.value, float)
```

**Parameters:**
- `max_difference`: Maximum difference for similarity (None = exact match)
- `coerce_type`: Coerce to int/float based on field type (default: `True`)

**JSON Number Handling:**

JSON (RFC 8259) has a single `Number` type, so `10` and `10.0` are equivalent.
Python distinguishes `int` and `float`. `NumericSpec` with `coerce_type=True`
automatically converts based on the declared field type:

- `age: int = NumericSpec()` → `30.0` becomes `30` (int)
- `score: float = NumericSpec()` → `30` becomes `30.0` (float)

### BooleanSpec

Optimized for boolean fields.

**Features:**
- Exact similarity matching
- Converts various values to bool

**Example:**
```python
from cobjectric import BaseModel
from cobjectric.specs import BooleanSpec

class Person(BaseModel):
    is_active: bool = BooleanSpec()
    is_verified: bool | None = BooleanSpec()

person1 = Person(is_active=True)
person2 = Person(is_active=1)  # Converts to True
person3 = Person(is_verified=None)  # Preserves None
```

### DatetimeSpec

Optimized for datetime string fields (ISO format).

**Features:**
- Datetime similarity with optional time difference tolerance
- Automatic normalization to ISO format
- Supports multiple input formats (auto-detect or custom)
- Converts datetime objects to ISO strings
- Preserves timezone information (Z suffix, offsets)

**Example:**
```python
from datetime import timedelta

from cobjectric import BaseModel
from cobjectric.specs import DatetimeSpec

class Event(BaseModel):
    # Auto-detect format
    created_at: str = DatetimeSpec()
    
    # Custom format
    updated_at: str = DatetimeSpec(format="%Y-%m-%d %H:%M:%S")
    
    # With tolerance
    timestamp: str = DatetimeSpec(max_difference=timedelta(hours=1))

event1 = Event(created_at="2024-01-15T10:30:00Z")
assert event1.fields.created_at.value == "2024-01-15T10:30:00Z"

event2 = Event(updated_at="2024-01-15 10:30:00")
assert event2.fields.updated_at.value == "2024-01-15T10:30:00"
```

**Parameters:**
- `format`: Optional datetime format string for parsing (auto-detect if None)
  - If specified, uses `datetime.strptime()` with this format
  - Example: `"%Y-%m-%d %H:%M:%S"`, `"%d/%m/%Y"`, etc.
  - If None, auto-detects common ISO formats
- `max_difference`: Maximum time difference as timedelta (None = exact match)
  - Use `timedelta(days=1)`, `timedelta(hours=1)`, `timedelta(minutes=30)`, etc.

## Using Custom Parameters

All Specs accept the same parameters as `Spec()`, allowing you to customize
weights, metadata, and other settings:

```python
from cobjectric.specs import TextSpec, NumericSpec

class Person(BaseModel):
    name: str = TextSpec(
        lower=False,
        similarity_weight=2.0,
        metadata={"description": "Full name"},
    )
    age: int = NumericSpec(
        max_difference=1.0,
        fill_rate_weight=1.5,
    )
```

## Contextual Normalizers

Some Specs (like `NumericSpec`) use **contextual normalizers** that receive
field context information. This allows intelligent type coercion based on
the declared field type.

### FieldContext

Contextual normalizers receive a `FieldContext` object with:
- `name`: The field name
- `field_type`: The declared Python type
- `spec`: The FieldSpec associated with the field

**Example:**
```python
from cobjectric import BaseModel, FieldContext, Spec

def my_contextual_normalizer(value: t.Any, context: FieldContext) -> t.Any:
    if context.field_type is int:
        return int(float(value))
    return value

class Person(BaseModel):
    age: int = Spec(normalizer=my_contextual_normalizer)
```

Normalizers can have two signatures:
- Simple: `Callable[[Any], Any]` (1 parameter)
- Contextual: `Callable[[Any, FieldContext], Any]` (2 parameters)

The system automatically detects which signature to use.

## Combining Specs

You can use multiple Specs together in the same model:

```python
from cobjectric import BaseModel
from cobjectric.specs import KeywordSpec, TextSpec, NumericSpec, BooleanSpec

class Person(BaseModel):
    id: str = KeywordSpec()
    name: str = TextSpec()
    age: int = NumericSpec()
    is_active: bool = BooleanSpec()

person = Person.from_dict({
    "id": "  ABC123  ",
    "name": "  JOHN DOE  ",
    "age": 30.0,
    "is_active": True,
})

assert person.fields.id.value == "ABC123"
assert person.fields.name.value == "john doe"
assert person.fields.age.value == 30
assert isinstance(person.fields.age.value, int)
```

## Related Topics

- [Field Specifications](field_specs.md) - Learn about the base `Spec()` function
- [Similarity](similarity.md) - Learn about similarity functions
- [Fill Rate](fill_rate.md) - Learn about fill rate functions

