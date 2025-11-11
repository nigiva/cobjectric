# Cobjectric

> [!WARNING]
> **Status**: 🚧 Work in Progress - This project is in early development

**Complex Object Metric** - A Python library for computing metrics on complex objects (JSON, dictionaries, lists, etc.).

[![Python Version](https://img.shields.io/badge/python-3.13.9-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 Description

Cobjectric is a library designed to help developers calculate metrics on complex objects such as JSON, dictionaries, and arrays. It was originally created for Machine Learning projects where comparing and evaluating generated JSON structures against ground truth data was a repetitive manual task.

## 🚀 Getting Started

### For Users

```bash
# TODO: Once published to PyPI
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
  version = {0.1.0}
}
```