# Cobjectric

**Complex Object Metric** - A Python library for computing metrics on complex objects (JSON, dictionaries, lists, etc.).

[![CI](https://github.com/nigiva/cobjectric/actions/workflows/ci.yml/badge.svg)](https://github.com/nigiva/cobjectric/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/nigiva/cobjectric/graph/badge.svg?token=8W3KJU8JG1)](https://codecov.io/github/nigiva/cobjectric)
[![PyPI version](https://img.shields.io/pypi/v/cobjectric.svg)](https://pypi.org/project/cobjectric/)
[![PyPI downloads](https://img.shields.io/pypi/dm/cobjectric.svg)](https://pypi.org/project/cobjectric/)
[![Python Version](https://img.shields.io/pypi/pyversions/cobjectric.svg)](https://pypi.org/project/cobjectric/)
[![Documentation](https://img.shields.io/badge/docs-cobjectric.nigiva.com-blue)](https://cobjectric.nigiva.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## 📖 Description

Cobjectric is a library designed to help developers calculate metrics on complex objects such as JSON, dictionaries, and arrays. It was originally created for Machine Learning projects where comparing and evaluating generated JSON structures against ground truth data was a repetitive manual task.

## 📦 Installation

```bash
pip install cobjectric
```

## 🚀 Core Features

Cobjectric provides **three main functionalities** for analyzing complex structured data:

### 1. Fill Rate - Measure Data Completeness

Compute how "complete" your data is by measuring which fields are filled vs missing.

```python
from cobjectric import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str

person = Person.from_dict({
    "name": "John Doe",
    "age": 30,
    # email is missing
})

result = person.compute_fill_rate()
print(result.fields.name.value)   # 1.0 (present)
print(result.fields.age.value)    # 1.0 (present)
print(result.fields.email.value)  # 0.0 (missing)
print(result.mean())              # 0.667 (2 out of 3 fields filled)
```

**Use cases**: Data quality assessment, completeness scoring, field-level statistics.

### 2. Fill Rate Accuracy - Compare Completeness States

Compare the completeness of two models (got vs expected). **Focus on field state** (filled/missing), not on actual values.

```python
got = Person.from_dict({"name": "John", "age": 30})           # email missing
expected = Person.from_dict({"name": "Jane", "age": 25, "email": "jane@example.com"})

accuracy = got.compute_fill_rate_accuracy(expected)
print(accuracy.fields.name.value)   # 1.0 (both filled)
print(accuracy.fields.age.value)    # 1.0 (both filled)
print(accuracy.fields.email.value)  # 0.0 (got missing, expected filled)
print(accuracy.mean())              # 0.667 (2 out of 3 states match)
```

**Note**: Fill Rate Accuracy compares **state only** (field present/missing), not values. To validate actual values, use Similarity.

**Use cases**: Validation pipelines, comparing generated vs expected data structures, quality control.

### 3. Similarity - Compare Values with Fuzzy Matching

Compare field values between two models with support for **fuzzy text matching** via `rapidfuzz` and intelligent list alignment strategies.

```python
from cobjectric import BaseModel, Spec, ListCompareStrategy
from cobjectric.similarity import fuzzy_similarity_factory

class Person(BaseModel):
    name: str = Spec(similarity_func=fuzzy_similarity_factory("WRatio"))
    tags: list[Tag] = Spec(list_compare_strategy=ListCompareStrategy.OPTIMAL_ASSIGNMENT)

got = Person.from_dict({"name": "John Doe", "tags": [...]})
expected = Person.from_dict({"name": "john doe", "tags": [...]})

similarity = got.compute_similarity(expected)
print(similarity.fields.name.value)  # 0.99 (fuzzy match despite case difference)
print(similarity.fields.tags.mean()) # Uses optimal assignment for best matching
```

**Key features**:
- **Fuzzy text matching** via `rapidfuzz`: handles typos, case differences, word order
- **List alignment strategies**:
  - `PAIRWISE`: Compare by index (default)
  - `LEVENSHTEIN`: Order-preserving alignment based on similarity
  - `OPTIMAL_ASSIGNMENT`: Hungarian algorithm for best one-to-one matching
- **Numeric similarity**: Gradual similarity based on difference thresholds

**Use cases**: ML model evaluation, fuzzy matching, comparing generated text with ground truth, list item matching.

### Additional Features

- **Pre-defined Specs**: Optimized Specs for common types (`KeywordSpec`, `TextSpec`, `NumericSpec`, `BooleanSpec`, `DatetimeSpec`)
- **Contextual Normalizers**: Normalizers that receive field context for intelligent type coercion
- **Statistical Aggregation**: `mean()`, `std()`, `var()`, `min()`, `max()`, `quantile()` on all results
- **Nested Models**: Recursive computation on complex structures
- **List Aggregation**: Access aggregated statistics across list items via `items.aggregated_fields.name.mean()`
- **Path Access**: `result["address.city"]` or `result["items[0].name"]`
- **Custom Functions**: Define your own fill rate, accuracy, or similarity functions per field
- **Field Normalizers**: Transform values before validation

See the [documentation](https://cobjectric.nigiva.com) for complete details.

## 📚 Full Documentation

**📖 [https://cobjectric.nigiva.com](https://cobjectric.nigiva.com)**

The documentation includes:

- [Quick Start](https://cobjectric.nigiva.com/quickstart/) - Get started in 5 minutes
- [Examples](https://cobjectric.nigiva.com/examples/) - Real-world usage examples
- [Complete API Reference](https://cobjectric.nigiva.com/api-reference/) - All classes and functions
- [Feature Guides](https://cobjectric.nigiva.com/) - In-depth guides for all features


## 🛠️ Development

### Getting Started

**Prerequisites**

- Python 3.13.9 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

1. Install dependencies with uv (including optional extras for testing):

```bash
uv sync --dev --all-extras
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
  version = {3.0.0}
}
```
