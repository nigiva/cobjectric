# Cobjectric

> [!WARNING] > **Status**: 🚧 Work in Progress - This project is in early development

**Complex Object Metric** - A Python library for computing metrics on complex objects (JSON, dictionaries, lists, etc.).

[![CI](https://github.com/nigiva/cobjectric/actions/workflows/ci.yml/badge.svg)](https://github.com/nigiva/cobjectric/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/nigiva/cobjectric/graph/badge.svg?token=8W3KJU8JG1)](https://codecov.io/github/nigiva/cobjectric)
[![PyPI version](https://img.shields.io/pypi/v/cobjectric.svg)](https://pypi.org/project/cobjectric/)
[![Python Version](https://img.shields.io/pypi/pyversions/cobjectric.svg)](https://pypi.org/project/cobjectric/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## 📖 Description

Cobjectric is a library designed to help developers calculate metrics on complex objects such as JSON, dictionaries, and arrays. It was originally created for Machine Learning projects where comparing and evaluating generated JSON structures against ground truth data was a repetitive manual task.

## 🚀 Getting Started

### For Users

```bash
pip install cobjectric
```

### For Development

**Prerequisites**

- Python 3.13.9 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

1. Install dependencies with uv:

```bash
uv sync --dev
```

2. Install pre-commit hooks:

```bash
uv run pre-commit install --hook-type pre-push
```

## 🛠️ Development

### Available Commands

The project uses [invoke](https://www.pyinvoke.org/) for task management.

To see all available commands:

```bash
uv run inv --list
# or shorter:
uv run inv -l
```

To get help on a specific command:

```bash
uv run inv --help <command>
# Example:
uv run inv --help precommit
```

## 📚 Usage

### Quick Example

Define a model with typed fields:

```python
from cobjectric import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str
    is_active: bool

# Create an instance
person = Person(
    name="John Doe",
    age=30,
    email="john.doe@example.com",
    is_active=True,
)

# Access fields
print(person.fields.name.value)   # "John Doe"
print(person.fields.age.value)    # 30
```

You can also create instances from dictionaries:

```python
# Create from dictionary
person = Person.from_dict({
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
    "is_active": True,
})
```

### Nested Models

You can define nested models by using another `BaseModel` subclass as a field type:

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

# Access nested model fields
print(person.fields.address.fields.street.value)  # "123 Main St"
print(person.fields.address.fields.city.value)    # "Anytown"
```

### List Fields

You can also define fields that contain lists of values:

```python
from cobjectric import BaseModel

class Experience(BaseModel):
    title: str
    company: str
    start_date: str
    end_date: str

class Person(BaseModel):
    name: str
    age: int
    email: str
    is_active: bool
    skills: list[str]
    experiences: list[Experience]

# Create from dictionary with list fields
person = Person.from_dict({
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
    "is_active": True,
    "skills": ["Python", "JavaScript", "Rust"],
    "experiences": [
        {
            "title": "Software Engineer",
            "company": "Tech Corp",
            "start_date": "2020-01-01",
            "end_date": "2022-01-01",
        },
    ],
})

# Access list fields
print(person.fields.skills.value)  # ["Python", "JavaScript", "Rust"]
print(person.fields.experiences.value[0].fields.title.value)  # "Software Engineer"
```

### Optional Fields and Union Types

You can define optional fields and union types:

```python
from cobjectric import BaseModel

class Person(BaseModel):
    name: str
    email: str | None  # Optional field
    id: str | int      # Union type
    scores: dict[str, int]  # Typed dict

# Create from dictionary
person = Person.from_dict({
    "name": "John Doe",
    "email": "john.doe@example.com",  # or None
    "id": 123,  # or "abc123"
    "scores": {"math": 90, "english": 85},
})

# Access fields
print(person.fields.email.value)  # "john.doe@example.com" or None
print(person.fields.id.value)   # 123
print(person.fields.scores.value)  # {"math": 90, "english": 85}
```

### Field Specifications (Spec)

You can add metadata to fields using the `Spec()` function:

```python
from cobjectric import BaseModel, Spec

class Person(BaseModel):
    name: str = Spec(metadata={"description": "The name of the person"})
    age: int = Spec()
    email: str
    is_active: bool

person = Person.from_dict({
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
    "is_active": True,
})

# Access field specifications
print(person.fields.name.spec.metadata)
# {"description": "The name of the person"}

# Fields without Spec() have default FieldSpec with empty metadata
print(person.fields.email.spec.metadata)
# {}
```

### Features

- **Typed Fields**: Define fields with type annotations
- **Field Specifications**: Add metadata to fields using `Spec()` function
- **Optional Fields**: Support for optional fields using `str | None` or `t.Optional[str]`
- **Union Types**: Support for union types like `str | int` or `t.Union[str, int]`
- **Typed Dicts**: Support for typed dictionaries with `dict[str, int]` syntax
  - **Partial Filtering**: Invalid entries are automatically filtered out, valid entries are kept
  - **Recursive Validation**: Deep validation for nested types like `dict[str, dict[str, int]]`
- **List Fields**: Support for list types with single element types (e.g., `list[str]`, `list[MyModel]`)
  - **Partial Filtering**: Invalid elements are automatically filtered out, valid elements are kept
  - **Recursive Validation**: Deep validation for nested types like `list[dict[str, int]]`
- **Nested Models**: Support for nested model structures
- **Type Validation**: Fields with invalid types are marked as missing or filtered out
- **Readonly Models**: Model instances are immutable after creation
- **Easy Field Access**: Access fields via the `.fields` attribute

### Documentation

For more information, see the [documentation](docs/index.md):

- [Quick Start](docs/quickstart.md) - Get started in 5 minutes
- [BaseModel and Fields](docs/base_model.md) - Detailed guide and API reference

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citing Cobjectric

If you use Cobjectric in your research or projects, please consider citing it:

```bibtex
@software{cobjectric2025,
  author = {Nigiva},
  title = {Cobjectric: A Library for Computing Metrics on Complex Objects},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nigiva/cobjectric}},
  version = {1.0.1}
}
```
