# Quick Start

Get up and running with Cobjectric in 5 minutes.

## Installation

```bash
pip install cobjectric
```

## Basic Usage

### Creating a Model

Define a model by subclassing `BaseModel` with typed fields:

```python
from cobjectric import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str
    is_active: bool
```

### Instantiating a Model

Create an instance by passing field values as keyword arguments:

```python
person = Person(
    name="John Doe",
    age=30,
    email="john.doe@example.com",
    is_active=True,
)
```

### Accessing Fields

Access fields through the `.fields` attribute:

```python
print(person.fields.name.value)  # "John Doe"
print(person.fields.age.value)   # 30
```

### Handling Missing Fields

If a required field is not provided, it will have `MissingValue`:

```python
from cobjectric import MissingValue

person = Person(name="Jane Doe", age=25)
print(person.fields.email.value is MissingValue)  # True
```

### Type Validation

If a field receives a value of the wrong type, it will also have `MissingValue`:

```python
person = Person(name="John", age="invalid")
print(person.fields.age.value is MissingValue)  # True
```

### Readonly Access

Model instances are immutable after creation. Attempting to modify a field will raise an error:

```python
person.name = "Jane"  # AttributeError
```

## Next Steps

- Learn more about [BaseModel and Fields](base_model.md)
- Check out the [API Reference](base_model.md#api-reference)

