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

**TODO**: Add usage examples and API documentation

```python
# Example usage will go here
from cobjectric import ...

# TODO: Add code examples
```

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
