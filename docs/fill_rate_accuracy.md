# Fill Rate Accuracy

Fill Rate Accuracy measures how well a "got" object matches an "expected" object in terms of **field completeness states** (filled vs missing). Unlike [Similarity](similarity.md), it focuses on whether fields are present, not their actual values.

## Concept

When comparing two objects:
- **Fill Rate Accuracy** = `1.0` if both have the same state for a field (both filled or both missing)
- **Fill Rate Accuracy** = `0.0` if states differ (one filled, one missing)

This is useful for validating that your data generation pipeline produces the right "shape" of data, regardless of the actual values.

## Basic Usage

```python
from cobjectric import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str

# Generated data (missing email)
got = Person.from_dict({"name": "John", "age": 30})

# Expected data (has email)
expected = Person.from_dict({
    "name": "Jane",
    "age": 25,
    "email": "jane@example.com"
})

accuracy = got.compute_fill_rate_accuracy(expected)
print(accuracy.fields.name.value)   # 1.0 (both filled)
print(accuracy.fields.age.value)    # 1.0 (both filled)
print(accuracy.fields.email.value)  # 0.0 (got missing, expected filled)
print(accuracy.mean())              # 0.667 (2 out of 3 match)
```

**Note**: The actual values of `name` and `age` don't matter. Fill Rate Accuracy only cares about field presence/absence.

## Key Differences

### Fill Rate Accuracy vs Similarity

| Aspect | Fill Rate Accuracy | Similarity |
|--------|-------------------|-----------|
| **Focuses on** | Field state (present/missing) | Field values |
| **Use case** | Validating data shape | Validating data quality |
| **Example** | Checks if email field exists | Checks if email value is correct |

### Example Comparison

```python
got = Person.from_dict({"name": "John Doe", "age": 30, "email": "john@example.com"})
expected = Person.from_dict({"name": "jane doe", "age": 30, "email": "jane@example.com"})

# Fill Rate Accuracy (comparing states)
accuracy = got.compute_fill_rate_accuracy(expected)
print(accuracy.fields.name.value)     # 1.0 (both fields filled)

# Similarity (comparing values)
similarity = got.compute_similarity(expected)
print(similarity.fields.name.value)   # ~0.92 (fuzzy match: "john" vs "jane")
```

## Use Cases

1. **Data Pipeline Validation**: Ensure your data generation produces complete objects
2. **Quality Control**: Verify that all required fields are populated
3. **Schema Conformance**: Check that generated data matches the expected structure
4. **Testing**: Assert that generated data has the correct shape

## Advanced Usage

You can combine Fill Rate Accuracy with other metrics:

```python
# Check data shape
accuracy = got.compute_fill_rate_accuracy(expected)
if accuracy.mean() < 0.8:
    print("Warning: Data shape doesn't match!")

# Then check data quality
similarity = got.compute_similarity(expected)
if similarity.mean() < 0.9:
    print("Warning: Data values don't match!")
```

See [Similarity](similarity.md) for value-based comparison and [Field Specifications](field_specs.md) for custom validation logic.

## API Reference

See the [API Reference](reference.md#fill-rate-accuracy-results) for the complete Fill Rate Accuracy result classes and methods.

