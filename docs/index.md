# Cobjectric Documentation

Welcome to the Cobjectric documentation. This guide will help you get started with using Cobjectric in your projects.

## Table of Contents

### Getting Started

1. [Quick Start](quickstart.md) - Get up and running in 5 minutes

### Core Concepts

2. [BaseModel](base_model.md) - Learn about the base model class, fields, and basic usage
3. [Field Types](field_types.md) - Learn about different field types (Optional, Union, Dict, List)
4. [Nested Models](nested_models.md) - Learn about nested model structures

### Advanced Features

5. [Field Specifications](field_specs.md) - Learn about Spec(), metadata, and field normalizers
6. [Fill Rate](fill_rate.md) - Learn about fill rate and fill rate accuracy computation
7. [Similarity](similarity.md) - Learn about similarity computation
8. [List Comparison Strategies](list_comparison.md) - Learn about strategies for comparing list[BaseModel] fields
9. [Path Access](path_access.md) - Learn about accessing fields by path notation

### Examples

9. [Examples](examples.md) - Practical examples demonstrating various features

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
