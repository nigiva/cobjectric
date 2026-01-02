# Cobjectric Documentation

Welcome to the Cobjectric documentation. This guide will help you get started with using Cobjectric in your projects.

## Table of Contents

### Getting Started

1. [Quick Start](quickstart.md) - Get up and running in 5 minutes, computing metrics on your data

### Model Fundamentals

2. [BaseModel](base_model.md) - Learn about the base model class, fields, and basic usage
3. [Field Types](field_types.md) - Learn about different field types (Optional, Union, Dict, List)
4. [Nested Models](nested_models.md) - Learn about nested model structures
5. [Path Access](path_access.md) - Learn about accessing fields by path notation

### Features

6. [Fill Rate](fill_rate.md) - Measure data completeness by checking which fields are filled vs missing
7. [Fill Rate Accuracy](fill_rate_accuracy.md) - Compare field states (filled/missing) between two objects
8. [Similarity](similarity.md) - Compare field values with fuzzy matching and advanced strategies
9. [Field Specifications](field_specs.md) - Learn about Spec(), metadata, and field normalizers
10. [Pre-defined Specs](specs.md) - Learn about KeywordSpec, TextSpec, NumericSpec, and other pre-defined Specs
11. [List Comparison Strategies](list_comparison.md) - Learn about strategies for comparing list[BaseModel] fields

### Examples & API

12. [Examples](examples.md) - Practical examples demonstrating various features
13. [API Reference](reference.md) - Complete API documentation of all classes and functions

## Overview

Cobjectric is a Python library for defining and managing complex object models with typed fields. It provides a clean, intuitive API for defining models and accessing their fields.

## Key Features

- **Typed Fields**: Define fields with type annotations
- **Type Validation**: Fields with invalid types are marked as missing
- **Readonly Access**: Model instances are immutable after creation
- **Easy Field Access**: Access fields via `.fields` attribute
- **Nested Models**: Support for complex nested structures
- **Fill Rate**: Measure data completeness and quality
- **Similarity**: Compare models and compute similarity scores
- **Path Access**: Access nested fields using path notation

## Getting Started

To get started, see the [Quick Start Guide](quickstart.md).
