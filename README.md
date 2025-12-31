# Cobjectric

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

## 📚 Usage

```bash
pip install cobjectric
```

### Quick Example

Cobjectric allows you to define typed models and compute **fill rate** metrics to measure data completeness:

```python
from cobjectric import BaseModel, Spec

class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: len(x) / 100)
    age: int
    email: str

# Create from dictionary
person = Person.from_dict({
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
})

# Compute fill rate (scoring)
result = person.compute_fill_rate()

print(result.fields.name.value)   # 0.08 (len("John Doe") = 8, 8/100)
print(result.fields.age.value)    # 1.0 (present)
print(result.fields.email.value)  # 1.0 (present)
print(result.mean())              # 0.693... (average fill rate)
```

Fill rate measures how "complete" each field is (0.0 = missing, 1.0 = present). You can define custom fill rate functions or use the default (0.0 for missing, 1.0 for present).

See the [documentation](docs/base_model.md) for more details on nested models, list fields, normalizers, and advanced features.

### Features

- **Fill Rate Scoring**: Compute completeness metrics (fill rate) for all fields with statistical aggregation
- **Fill Rate Accuracy**: Compare field completeness between two models (got vs expected)
- **Typed Models**: Define models with type annotations and automatic validation
- **Nested Models**: Support for nested model structures with recursive fill rate computation
- **Path Access**: Access fields using path notation like `result["address.city"]`
- **Field Normalizers**: Transform field values before validation
- **Flexible Types**: Support for optional fields, union types, typed dicts, and lists

For complete feature list and details, see the [documentation](docs/base_model.md).

### Documentation

For more information, see the [documentation](docs/index.md):

- [Quick Start](docs/quickstart.md) - Get started in 5 minutes
- [BaseModel and Fields](docs/base_model.md) - Detailed guide and API reference


## 🛠️ Development

### Getting Started

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

### Release Guide

See the [RELEASE.md](RELEASE.md) file for the release guide.

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
  version = {1.1.0}
}
```
