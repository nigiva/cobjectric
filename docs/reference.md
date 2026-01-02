# API Reference

Complete API documentation for Cobjectric.

## Core Model Classes

See [BaseModel Guide](base_model.md) for detailed usage examples.

:::cobjectric.BaseModel
    options:
      show_source: true
      docstring_style: google

:::cobjectric.FieldSpec
    options:
      show_source: true
      docstring_style: google

:::cobjectric.Spec
    options:
      show_source: true
      docstring_style: google

See [Field Specifications Guide](field_specs.md) for advanced customization.

## Sentinel Values

:::cobjectric.MissingValue
    options:
      show_source: true
      docstring_style: google

## Context

:::cobjectric.FieldContext
    options:
      show_source: true
      docstring_style: google

## Field Results

See [Fill Rate Guide](fill_rate.md) for usage and examples.

### Fill Rate Results

:::cobjectric.FillRateFieldResult
    options:
      show_source: true
      docstring_style: google

:::cobjectric.FillRateFieldCollection
    options:
      show_source: true
      docstring_style: google

:::cobjectric.FillRateModelResult
    options:
      show_source: true
      docstring_style: google

### Fill Rate Accuracy Results

See [Fill Rate Accuracy Guide](fill_rate_accuracy.md) for usage and examples.

:::cobjectric.FillRateAccuracyFieldResult
    options:
      show_source: true
      docstring_style: google

:::cobjectric.FillRateAccuracyFieldCollection
    options:
      show_source: true
      docstring_style: google

:::cobjectric.FillRateAccuracyModelResult
    options:
      show_source: true
      docstring_style: google

### Similarity Results

See [Similarity Guide](similarity.md) for fuzzy matching and advanced strategies.

:::cobjectric.SimilarityFieldResult
    options:
      show_source: true
      docstring_style: google

:::cobjectric.SimilarityFieldCollection
    options:
      show_source: true
      docstring_style: google

:::cobjectric.SimilarityModelResult
    options:
      show_source: true
      docstring_style: google

## List Results

See [List Comparison Strategies Guide](list_comparison.md) for list handling and aggregation.

:::cobjectric.FillRateListResult
    options:
      show_source: true
      docstring_style: google

:::cobjectric.FillRateAggregatedFieldResult
    options:
      show_source: true
      docstring_style: google

:::cobjectric.FillRateAggregatedFieldCollection
    options:
      show_source: true
      docstring_style: google

:::cobjectric.FillRateAggregatedModelResult
    options:
      show_source: true
      docstring_style: google

:::cobjectric.FillRateNestedListAggregatedResult
    options:
      show_source: true
      docstring_style: google

## Decorator Info Classes

:::cobjectric.FillRateFuncInfo
    options:
      show_source: true
      docstring_style: google

:::cobjectric.FillRateAccuracyFuncInfo
    options:
      show_source: true
      docstring_style: google

:::cobjectric.SimilarityFuncInfo
    options:
      show_source: true
      docstring_style: google

## List Comparison Strategies

:::cobjectric.ListCompareStrategy
    options:
      show_source: true
      docstring_style: google

## Similarity Functions

See [Similarity Guide](similarity.md) for implementation details and best practices.

:::cobjectric.exact_similarity
    options:
      show_source: true
      docstring_style: google

:::cobjectric.fuzzy_similarity_factory
    options:
      show_source: true
      docstring_style: google

:::cobjectric.numeric_similarity_factory
    options:
      show_source: true
      docstring_style: google

:::cobjectric.datetime_similarity_factory
    options:
      show_source: true
      docstring_style: google

## Pre-defined Specs

See [Pre-defined Specs Guide](specs.md) for detailed usage and recommendations.

### KeywordSpec

:::cobjectric.KeywordSpec
    options:
      show_source: true
      docstring_style: google

### TextSpec

:::cobjectric.TextSpec
    options:
      show_source: true
      docstring_style: google

### NumericSpec

:::cobjectric.NumericSpec
    options:
      show_source: true
      docstring_style: google

### BooleanSpec

:::cobjectric.BooleanSpec
    options:
      show_source: true
      docstring_style: google

### DatetimeSpec

:::cobjectric.DatetimeSpec
    options:
      show_source: true
      docstring_style: google

## Default Functions

### Fill Rate Functions

:::cobjectric.not_missing_fill_rate
    options:
      show_source: true
      docstring_style: google

### Fill Rate Accuracy Functions

:::cobjectric.same_state_fill_rate_accuracy
    options:
      show_source: true
      docstring_style: google

## Decorator Functions

:::cobjectric.fill_rate_func
    options:
      show_source: true
      docstring_style: google

:::cobjectric.fill_rate_accuracy_func
    options:
      show_source: true
      docstring_style: google

:::cobjectric.similarity_func
    options:
      show_source: true
      docstring_style: google

:::cobjectric.field_normalizer
    options:
      show_source: true
      docstring_style: google

## Exceptions

All exceptions inherit from `CobjectricError`. Handle them appropriately in your error handling logic.

### Base Exception

:::cobjectric.CobjectricError
    options:
      show_source: true
      docstring_style: google

### Type Exceptions

:::cobjectric.UnsupportedTypeError
    options:
      show_source: true
      docstring_style: google

:::cobjectric.UnsupportedListTypeError
    options:
      show_source: true
      docstring_style: google

:::cobjectric.MissingListTypeArgError
    options:
      show_source: true
      docstring_style: google

### Function Definition Exceptions

:::cobjectric.DuplicateFillRateFuncError
    options:
      show_source: true
      docstring_style: google

:::cobjectric.DuplicateFillRateAccuracyFuncError
    options:
      show_source: true
      docstring_style: google

:::cobjectric.DuplicateSimilarityFuncError
    options:
      show_source: true
      docstring_style: google

### Validation Exceptions

:::cobjectric.InvalidFillRateValueError
    options:
      show_source: true
      docstring_style: google

:::cobjectric.InvalidWeightError
    options:
      show_source: true
      docstring_style: google

:::cobjectric.InvalidAggregatedFieldError
    options:
      show_source: true
      docstring_style: google

:::cobjectric.InvalidListCompareStrategyError
    options:
      show_source: true
      docstring_style: google
