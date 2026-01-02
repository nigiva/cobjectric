# Fill Rate

Fill rate is a metric that measures how "complete" or "filled" a field value is. It's a float between 0.0 and 1.0, where:
- `0.0` means the field is completely empty or missing
- `1.0` means the field is completely filled
- Values in between represent partial completeness

Fill rate is particularly useful for data quality assessment, where you want to measure how complete your data is across multiple fields.

## Using Spec(fill_rate_func=...)

You can define a custom fill rate function directly in the `Spec()` function:

```python
from cobjectric import BaseModel, Spec

class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: len(x) / 100)
    age: int
    email: str

person = Person(name="John Doe", age=30, email="john@example.com")
result = person.compute_fill_rate()

print(result.fields.name.value)  # 0.08 (len("John Doe") = 8, 8/100 = 0.08)
print(result.fields.age.value)   # 1.0 (default: present = 1.0)
print(result.fields.email.value) # 1.0 (default: present = 1.0)
```

## Using @fill_rate_func Decorator

You can also define fill rate functions using the `@fill_rate_func` decorator:

```python
from cobjectric import BaseModel, fill_rate_func, MissingValue
import typing as t

class Person(BaseModel):
    name: str
    email: str
    age: int

    @fill_rate_func("name", "email")
    def fill_rate_name_email(x: t.Any) -> float:
        return len(x) / 100 if x is not MissingValue else 0.0

    @fill_rate_func("age")
    def fill_rate_age(x: t.Any) -> float:
        return 1.0 if x is not MissingValue else 0.0

person = Person(name="John", email="john@example.com", age=30)
result = person.compute_fill_rate()

print(result.fields.name.value)  # 0.04 (len("John") = 4, 4/100 = 0.04)
print(result.fields.email.value) # 0.16 (len("john@example.com") = 16, 16/100 = 0.16)
print(result.fields.age.value)   # 1.0
```

## Pattern Matching

The `@fill_rate_func` decorator supports glob patterns to match multiple fields:

```python
class Person(BaseModel):
    name_first: str
    name_last: str
    age: int

    @fill_rate_func("name_*")
    def fill_rate_name_fields(x: t.Any) -> float:
        return len(x) / 100 if x is not MissingValue else 0.0

person = Person(name_first="John", name_last="Doe", age=30)
result = person.compute_fill_rate()

print(result.fields.name_first.value)  # 0.04
print(result.fields.name_last.value)   # 0.03
print(result.fields.age.value)        # 1.0
```

## Default Fill Rate Function

If no `fill_rate_func` is specified, the default behavior is:
- Returns `0.0` if the field value is `MissingValue`
- Returns `1.0` otherwise

## Fill Rate Validation

Fill rate functions must return a float (or int convertible to float) between 0.0 and 1.0. If a function returns an invalid value, `InvalidFillRateValueError` is raised:

```python
from cobjectric import InvalidFillRateValueError

class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: 1.5)  # Invalid: > 1.0

person = Person(name="John")
try:
    result = person.compute_fill_rate()
except InvalidFillRateValueError as e:
    print(f"Error: {e}")
```

## Duplicate Fill Rate Functions

A field can only have one `fill_rate_func`. If multiple functions are defined (via `Spec()` and `@fill_rate_func`, or multiple decorators), `DuplicateFillRateFuncError` is raised:

```python
from cobjectric import DuplicateFillRateFuncError

class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: 0.5)

    @fill_rate_func("name")
    def fill_rate_name(x: t.Any) -> float:
        return 0.6

person = Person(name="John")
try:
    result = person.compute_fill_rate()
except DuplicateFillRateFuncError as e:
    print(f"Error: {e}")
```

## Fill Rate on Normalized Values

Fill rate functions are applied to the **normalized value** (after normalizers have been applied):

```python
class Person(BaseModel):
    name: str = Spec(
        normalizer=lambda x: x.lower(),
        fill_rate_func=lambda x: len(x) / 100,
    )

person = Person(name="JOHN DOE")
result = person.compute_fill_rate()

# Normalized value is "john doe" (len=8), not "JOHN DOE" (len=8)
print(result.fields.name.value)  # 0.08
```

## Computing Fill Rate

To compute fill rate for all fields in a model, call the `compute_fill_rate()` method:

```python
class Person(BaseModel):
    name: str
    age: int
    email: str

person = Person(name="John Doe", age=30)
result = person.compute_fill_rate()

print(result.fields.name.value)  # 1.0
print(result.fields.age.value)   # 1.0
print(result.fields.email.value) # 0.0 (missing)
```

## ModelResult

The `compute_fill_rate()` method returns a `ModelResult` object that provides:

- **Field access**: `result.fields.name` returns a `FieldResult` with the fill rate value and weight
- **Statistical methods**: `mean()` (weighted), `max()`, `min()`, `std()`, `var()`, `quantile(q)`

```python
class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: 0.3)
    age: int = Spec(fill_rate_func=lambda x: 0.8)
    email: str = Spec(fill_rate_func=lambda x: 0.5)

person = Person(name="John", age=30, email="john@example.com")
result = person.compute_fill_rate()

print(result.mean())      # 0.533... (weighted average of 0.3, 0.8, 0.5)
print(result.max())       # 0.8
print(result.min())       # 0.3
print(result.std())       # Standard deviation
print(result.var())       # Variance
print(result.quantile(0.25))  # 25th percentile
print(result.quantile(0.50))  # 50th percentile (median)
print(result.quantile(0.75))  # 75th percentile
```

**Note**: The `mean()` method calculates a **weighted mean** if weights are specified (see [Weighted Mean](#weighted-mean) section below). By default, all fields have weight `1.0`, so the mean is equivalent to a simple average.

## Weighted Mean

By default, all fields have a weight of `1.0`, which means the mean is calculated as a simple average. However, you can assign different weights to fields to calculate a **weighted mean**. This is useful when some fields are more important than others in your data quality assessment.

### Using Weight in Spec()

You can set a weight directly in `Spec()`:

```python
class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: 0.5, weight=2.0)
    age: int = Spec(fill_rate_func=lambda x: 1.0, weight=1.0)

person = Person(name="John", age=30)
result = person.compute_fill_rate()

# Weighted mean: (0.5 * 2.0 + 1.0 * 1.0) / (2.0 + 1.0) = 2.0 / 3.0 = 0.666...
print(result.mean())  # 0.666...
```

### Using Weight in Decorator

You can also set a weight in the `@fill_rate_func` decorator:

```python
class Person(BaseModel):
    name: str
    age: int

    @fill_rate_func("name", weight=2.0)
    def fill_rate_name(x: t.Any) -> float:
        return 0.5 if x is not MissingValue else 0.0

    @fill_rate_func("age", weight=1.0)
    def fill_rate_age(x: t.Any) -> float:
        return 1.0 if x is not MissingValue else 0.0

person = Person(name="John", age=30)
result = person.compute_fill_rate()

print(result.fields.name.weight)  # 2.0
print(result.fields.age.weight)   # 1.0
print(result.mean())              # 0.666... (weighted mean)
```

### Decorator Weight Overrides Spec Weight

If both `Spec(fill_rate_weight=...)` and `@fill_rate_func(..., weight=...)` are defined for the same field, the decorator weight takes precedence:

```python
class Person(BaseModel):
    name: str = Spec(fill_rate_weight=1.0)

    @fill_rate_func("name", weight=2.0)
    def fill_rate_name(x: t.Any) -> float:
        return 1.0 if x is not MissingValue else 0.0

person = Person(name="John")
result = person.compute_fill_rate()

# Decorator weight (2.0) overrides Spec weight (1.0)
print(result.fields.name.weight)  # 2.0
```

### Weight Validation

Weight must be `>= 0.0`. Negative weights will raise `InvalidWeightError`:

```python
from cobjectric import InvalidWeightError

# This will raise InvalidWeightError
try:
    Spec(weight=-1.0)
except InvalidWeightError as e:
    print(f"Error: {e}")

# This will also raise InvalidWeightError
try:
    @fill_rate_func("name", weight=-1.0)
    def fill_rate_name(x: t.Any) -> float:
        return 1.0
except InvalidWeightError as e:
    print(f"Error: {e}")
```

### Weighted Mean Formula

The weighted mean is calculated as:

```
weighted_mean = sum(value * weight) / sum(weight)
```

This means:
- You don't need weights to sum to 1.0
- Fields with higher weights have more influence on the mean
- Fields with weight `0.0` don't contribute to the mean (but are still included in max/min)
- If all weights sum to 0.0, the mean returns `0.0`

### Weighted Mean with Nested Models

Weights work recursively with nested models:

```python
class Address(BaseModel):
    street: str = Spec(fill_rate_func=lambda x: 0.5, weight=2.0)
    city: str = Spec(fill_rate_func=lambda x: 1.0, weight=1.0)

class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: 0.8, weight=1.0)
    address: Address

person = Person.from_dict({
    "name": "John",
    "address": {"street": "123 Main St", "city": "Anytown"},
})
result = person.compute_fill_rate()

# Weighted mean across all fields (including nested):
# (0.8 * 1.0 + 0.5 * 2.0 + 1.0 * 1.0) / (1.0 + 2.0 + 1.0) = 2.8 / 4.0 = 0.7
print(result.mean())  # 0.7
```

### Max and Min with Weights

The `max()` and `min()` methods are **not affected by weights** - they simply return the maximum and minimum fill rate values:

```python
class Person(BaseModel):
    name: str = Spec(fill_rate_func=lambda x: 0.3, fill_rate_weight=2.0)
    age: int = Spec(fill_rate_func=lambda x: 0.8, fill_rate_weight=0.5)

person = Person(name="John", age=30)
result = person.compute_fill_rate()

print(result.max())  # 0.8 (not affected by weights)
print(result.min())  # 0.3 (not affected by weights)
print(result.mean())  # 0.466... (weighted mean)
```

## Fill Rate with Nested Models

Fill rate computation works recursively with nested models:

```python
class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

person = Person.from_dict({
    "name": "John Doe",
    "address": {
        "street": "123 Main St",
        "city": "Anytown",
    },
})
result = person.compute_fill_rate()

# Access fill rate for top-level field
print(result.fields.name.value)  # 1.0

# Access fill rate for nested model (returns ModelResult)
nested_result = result.fields.address
print(nested_result.fields.street.value)  # 1.0
print(nested_result.fields.city.value)   # 1.0

# Statistical methods aggregate across all fields (including nested)
print(result.mean())  # 1.0 (all fields are present)
```

## Missing Nested Models

If a nested model is missing (has `MissingValue`), all its fields are considered to have a fill rate of `0.0`:

```python
class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

person = Person(name="John Doe")  # address is missing
result = person.compute_fill_rate()

print(result.fields.name.value)  # 1.0
nested_result = result.fields.address
print(nested_result.fields.street.value)  # 0.0
print(nested_result.fields.city.value)    # 0.0
```

## Fill Rate with List Fields

Fill rate computation supports list fields with two different behaviors depending on the list element type.

### List of Primitive Types

For `list[str]`, `list[int]`, etc., the fill rate is:
- `0.0` if the field is `MissingValue` or an empty list `[]`
- `1.0` if the list is non-empty

```python
class Person(BaseModel):
    name: str
    tags: list[str]

person = Person(name="John")
result = person.compute_fill_rate()

print(result.fields.name.value)  # 1.0
print(result.fields.tags.value)  # 0.0 (tags is MissingValue)

person = Person(name="John", tags=["python", "rust"])
result = person.compute_fill_rate()

print(result.fields.tags.value)  # 1.0 (tags is non-empty)
```

### List of BaseModel

For `list[BaseModel]`, you get a `ListResult` with two access modes:

**Access by Index** (individual item):

```python
class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    customer: str
    items: list[Item]

order = Order.from_dict({
    "customer": "John",
    "items": [
        {"name": "Apple", "price": 1.0},
        {"name": "Banana"},  # price missing
    ],
})

result = order.compute_fill_rate()

# Access list result
items_result = result.fields.items  # ListResult
print(len(items_result))  # 2

# Access individual item by index
item0_result = items_result[0]  # ModelResult for first item
print(item0_result.fields.name.value)   # 1.0
print(item0_result.fields.price.value)  # 1.0

item1_result = items_result[1]  # ModelResult for second item
print(item1_result.fields.name.value)   # 1.0
print(item1_result.fields.price.value)  # 0.0 (price is missing)

# Iterate over items
for item_result in items_result:
    print(item_result.mean())  # Mean fill rate for each item
```

**Aggregated Access** (statistics across all items):

```python
# Recommended: Use aggregated_fields for clarity
name_aggregated = result.fields.items.aggregated_fields.name  # AggregatedFieldResult
print(name_aggregated.values)  # [1.0, 1.0] - fill rate for name in each item
print(name_aggregated.mean())  # 1.0 - mean fill rate for name across items
print(name_aggregated.max())   # 1.0
print(name_aggregated.min())   # 1.0
print(name_aggregated.std())   # 0.0
print(name_aggregated.var())   # 0.0
print(name_aggregated.quantile(0.5))  # 1.0

price_aggregated = result.fields.items.aggregated_fields.price
print(price_aggregated.values)  # [1.0, 0.0]
print(price_aggregated.mean())  # 0.5
```

**Nested Models in Lists**:

```python
class Address(BaseModel):
    city: str
    street: str

class Item(BaseModel):
    name: str
    address: Address

class Order(BaseModel):
    items: list[Item]

order = Order.from_dict({
    "items": [
        {"name": "Item1", "address": {"city": "NYC", "street": "Main"}},
        {"name": "Item2", "address": {"city": "LA"}},  # street missing
    ],
})

result = order.compute_fill_rate()

# Access nested model through aggregated
address_aggregated = result.fields.items.aggregated_fields.address  # AggregatedModelResult
city_aggregated = address_aggregated.city  # AggregatedFieldResult (AggregatedModelResult uses __getattr__ directly)
print(city_aggregated.values)  # [1.0, 1.0]
print(address_aggregated.street.values)  # [1.0, 0.0]
```

**Limitations - Nested Lists**:

Aggregation only works for `list[BaseModel]`, not for nested lists like `list[list[BaseModel]]`:

```python
class Tag(BaseModel):
    name: str

class Item(BaseModel):
    tags: list[Tag]  # This works with aggregation

class Catalog(BaseModel):
    items: list[Item]

catalog = Catalog.from_dict({
    "items": [
        {"tags": [{"name": "a"}, {"name": "b"}]},
        {"tags": [{"name": "c"}]},
    ],
})

result = catalog.compute_fill_rate()

# Accessing items.aggregated_fields.tags returns the mean fill rate
# of each tags list, not individual tag fields
tags_agg = result.fields.items.aggregated_fields.tags
# Returns AggregatedFieldResult with values = [1.0, 1.0]
# (one value per item, representing the mean fill rate of each tags list)

# To access individual tag fields, use indexed access:
item0_tags = result.fields.items[0].fields.tags  # ListResult
tag0 = item0_tags[0]  # ModelResult for first tag
```

For `list[list[str]]` or `list[list[int]]`, the field is treated as a `list[Primitive]`:

```python
class Person(BaseModel):
    skills: list[list[str]]  # Nested list of primitives

person = Person(skills=[["Python", "Rust"], ["JavaScript"]])

result = person.compute_fill_rate()

# skills is treated as list[Primitive], returns FieldResult
# 0.0 if empty, 1.0 if non-empty
print(result.fields.skills.value)  # 1.0 (non-empty list)
```

:warning: No aggregation is available for nested lists of primitives

**Empty or Missing Lists**:

```python
# Empty list
order = Order.from_dict({"items": []})
result = order.compute_fill_rate()
print(len(result.fields.items))  # 0
print(result.fields.items.mean())  # 0.0

# Missing list
order = Order(customer="John")  # items is MissingValue
result = order.compute_fill_rate()
print(len(result.fields.items))  # 0
print(result.fields.items.mean())  # 0.0
```

# Fill Rate Accuracy

Fill rate accuracy measures whether fields are correctly filled compared to an expected model, regardless of the actual values. It's different from fill rate, which measures completeness. Fill rate accuracy checks if the "filled" or "missing" state matches between two models.

## Concept

Fill rate accuracy returns:
- `1.0` if both models have the same state (both filled or both missing)
- `0.0` if one is filled and the other is missing

| got | expected | accuracy |
|-----|----------|----------|
| filled | filled | 1.0 |
| MissingValue | MissingValue | 1.0 |
| filled | MissingValue | 0.0 |
| MissingValue | filled | 0.0 |

## Computing Fill Rate Accuracy

To compute fill rate accuracy, call the `compute_fill_rate_accuracy()` method with an expected model:

```python
class Person(BaseModel):
    name: str
    age: int
    email: str

person_got = Person.from_dict({"name": "John", "age": 30})
person_expected = Person.from_dict({"name": "Jane", "email": "jane@example.com"})

result = person_got.compute_fill_rate_accuracy(person_expected)

# name: both filled -> 1.0
print(result.fields.name.value)  # 1.0
# age: got filled, expected missing -> 0.0
print(result.fields.age.value)    # 0.0
# email: got missing, expected filled -> 0.0
print(result.fields.email.value)  # 0.0
```

## Using Spec(fill_rate_accuracy_func=...)

You can define a custom fill rate accuracy function in `Spec()`:

```python
class Person(BaseModel):
    name: str = Spec(
        fill_rate_accuracy_func=lambda got, exp: 0.8
        if (got is not MissingValue) == (exp is not MissingValue)
        else 0.0
    )
    age: int

person_got = Person(name="John", age=30)
person_expected = Person(name="Jane", age=25)

result = person_got.compute_fill_rate_accuracy(person_expected)
print(result.fields.name.value)  # 0.8 (custom function)
print(result.fields.age.value)   # 1.0 (default function)
```

## Using @fill_rate_accuracy_func Decorator

You can also use the `@fill_rate_accuracy_func` decorator:

```python
from cobjectric import BaseModel, fill_rate_accuracy_func
import typing as t

class Person(BaseModel):
    name: str
    email: str

    @fill_rate_accuracy_func("name", "email")
    def accuracy_name_email(got: t.Any, expected: t.Any) -> float:
        return 0.9 if (got is not MissingValue) == (expected is not MissingValue) else 0.0

person_got = Person(name="John", email="john@example.com")
person_expected = Person(name="Jane", email="jane@example.com")

result = person_got.compute_fill_rate_accuracy(person_expected)
print(result.fields.name.value)  # 0.9
print(result.fields.email.value) # 0.9
```

## Separate Weights for Fill Rate Accuracy

Fill rate accuracy uses its own weight system, separate from fill rate weights:

```python
class Person(BaseModel):
    name: str = Spec(
        fill_rate_func=lambda x: 0.5,
        fill_rate_weight=2.0,  # Weight for compute_fill_rate()
        fill_rate_accuracy_func=lambda got, exp: 1.0
        if (got is not MissingValue) == (exp is not MissingValue)
        else 0.0,
        fill_rate_accuracy_weight=1.5,  # Weight for compute_fill_rate_accuracy()
    )

person_got = Person(name="John")
person_expected = Person(name="Jane")

# Fill rate result uses fill_rate_weight
fill_rate_result = person_got.compute_fill_rate()
print(fill_rate_result.fields.name.weight)  # 2.0

# Accuracy result uses fill_rate_accuracy_weight
accuracy_result = person_got.compute_fill_rate_accuracy(person_expected)
print(accuracy_result.fields.name.weight)  # 1.5
```

## Fill Rate Accuracy with Nested Models

Fill rate accuracy works recursively with nested models:

```python
class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

person_got = Person.from_dict({
    "name": "John",
    "address": {"street": "123 Main St", "city": "Anytown"},
})
person_expected = Person.from_dict({
    "name": "Jane",
    "address": {"street": "456 Oak Ave", "city": "Somewhere"},
})

result = person_got.compute_fill_rate_accuracy(person_expected)

# Both have name -> 1.0
print(result.fields.name.value)  # 1.0

# Both have address with all fields -> 1.0 for nested fields
print(result.fields.address.fields.street.value)  # 1.0
print(result.fields.address.fields.city.value)   # 1.0
```

## Fill Rate Accuracy with List Fields

Fill rate accuracy supports list fields with different behaviors depending on the list element type.

### List of Primitive Types

For `list[str]`, `list[int]`, etc., accuracy compares whether both lists are filled or both are empty:

```python
class Person(BaseModel):
    tags: list[str]

person_got = Person(tags=["python", "rust"])
person_expected = Person(tags=["java", "go"])

result = person_got.compute_fill_rate_accuracy(person_expected)
print(result.fields.tags.value)  # 1.0 (both have non-empty lists)

person_got = Person(tags=["python"])
person_expected = Person()  # tags is MissingValue

result = person_got.compute_fill_rate_accuracy(person_expected)
print(result.fields.tags.value)  # 0.0 (one filled, one missing)
```

### List of BaseModel

For `list[BaseModel]`, accuracy compares items one by one. By default, items are compared pairwise (by index), but you can use different comparison strategies to align items when list order may differ.

**Using List Comparison Strategies**:

You can use `list_compare_strategy` to control how items are aligned:

```python
from cobjectric import BaseModel, Spec, ListCompareStrategy

class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    # Default: pairwise comparison (by index)
    items_pairwise: list[Item]
    # Levenshtein alignment (preserves relative order)
    items_levenshtein: list[Item] = Spec(
        list_compare_strategy=ListCompareStrategy.LEVENSHTEIN
    )
    # Optimal assignment (best matching regardless of order)
    items_optimal: list[Item] = Spec(
        list_compare_strategy=ListCompareStrategy.OPTIMAL_ASSIGNMENT
    )
```

For detailed information about all available strategies, see [List Comparison Strategies](list_comparison.md).

**Default Behavior (Pairwise)**:

By default, items are compared pairwise (by index). Only items that exist in both lists at the same position are compared:

```python
class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    items: list[Item]

order_got = Order.from_dict({
    "items": [
        {"name": "Apple", "price": 1.0},
        {"name": "Banana", "price": 0.5},
    ],
})
order_expected = Order.from_dict({
    "items": [
        {"name": "Orange", "price": 2.0},
        {"name": "Cherry", "price": 3.0},
    ],
})

result = order_got.compute_fill_rate_accuracy(order_expected)

# Both have 2 items, all fields filled -> accuracy = 1.0 for all
print(len(result.fields.items))  # 2
print(result.fields.items[0].fields.name.value)   # 1.0
print(result.fields.items[0].fields.price.value)  # 1.0

# Aggregated access works too (recommended API)
print(result.fields.items.aggregated_fields.name.values)   # [1.0, 1.0]
print(result.fields.items.aggregated_fields.price.values)  # [1.0, 1.0]
```

**Different List Lengths**:

When lists have different lengths, only the items that exist in both lists are compared (up to the minimum length):

```python
order_got = Order.from_dict({
    "items": [
        {"name": "Apple"},
    ],
})
order_expected = Order.from_dict({
    "items": [
        {"name": "Orange"},
        {"name": "Cherry"},
    ],
})

result = order_got.compute_fill_rate_accuracy(order_expected)

# Only first item is compared (min(len(got), len(expected)))
print(len(result.fields.items))  # 1
print(result.fields.items[0].fields.name.value)  # 1.0
```

## Duplicate Fill Rate Accuracy Functions

A field can only have one `fill_rate_accuracy_func`. If multiple functions are defined, `DuplicateFillRateAccuracyFuncError` is raised:

```python
from cobjectric import DuplicateFillRateAccuracyFuncError

class Person(BaseModel):
    name: str = Spec(fill_rate_accuracy_func=lambda got, exp: 0.5)

    @fill_rate_accuracy_func("name")
    def accuracy_name(got: t.Any, expected: t.Any) -> float:
        return 0.6

person_got = Person(name="John")
person_expected = Person(name="Jane")

try:
    result = person_got.compute_fill_rate_accuracy(person_expected)
except DuplicateFillRateAccuracyFuncError as e:
    print(f"Error: {e}")
```

## Related Topics

- [BaseModel](base_model.md) - Learn about the base model class
- [Field Types](field_types.md) - Learn about different field types
- [Nested Models](nested_models.md) - Learn about nested model structures
- [Field Specifications](field_specs.md) - Learn about Spec() and field normalizers
- [Similarity](similarity.md) - Learn about similarity computation
- [List Comparison Strategies](list_comparison.md) - Detailed guide on list comparison strategies

## API Reference

See the [API Reference](reference.md#field-results) for the complete Fill Rate result classes and methods.

