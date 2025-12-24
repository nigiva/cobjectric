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

### Features

- **Typed Fields**: Define fields with type annotations
- **Nested Models**: Support for nested model structures
- **Type Validation**: Fields with invalid types are marked as missing
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
