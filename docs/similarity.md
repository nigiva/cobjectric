# Similarity

Similarity measures how similar the **values** of fields are between two models. It's different from fill rate accuracy, which only checks if fields are filled or missing. Similarity compares the actual values and returns a score between 0.0 and 1.0.

## Concept

Similarity returns:
- `1.0` if values are equal
- `0.0` if values are different
- Values between 0.0 and 1.0 for partial similarity (when using custom similarity functions)

| got | expected | similarity |
|-----|----------|------------|
| "John" | "John" | 1.0 |
| "John" | "Jane" | 0.0 |
| MissingValue | MissingValue | 1.0 |
| "John" | MissingValue | 0.0 |
| MissingValue | "John" | 0.0 |

## Computing Similarity

To compute similarity, call the `compute_similarity()` method with an expected model:

```python
class Person(BaseModel):
    name: str
    age: int
    email: str

person_got = Person.from_dict({"name": "John", "age": 30})
person_expected = Person.from_dict({"name": "John", "email": "john@example.com"})

result = person_got.compute_similarity(person_expected)

# name: both same -> 1.0
print(result.fields.name.value)  # 1.0
# age: got filled, expected missing -> 0.0
print(result.fields.age.value)    # 0.0
# email: got missing, expected filled -> 0.0
print(result.fields.email.value)  # 0.0
```

## Using Spec(similarity_func=...)

You can define a custom similarity function in `Spec()`:

```python
class Person(BaseModel):
    name: str = Spec(
        similarity_func=lambda got, exp: 0.8 if got == exp else 0.0
    )
    age: int

person_got = Person(name="John", age=30)
person_expected = Person(name="John", age=25)

result = person_got.compute_similarity(person_expected)
print(result.fields.name.value)  # 0.8 (custom function)
print(result.fields.age.value)   # 0.0 (default function: different values)
```

## Using @similarity_func Decorator

You can also use the `@similarity_func` decorator:

```python
from cobjectric import BaseModel, similarity_func
import typing as t

class Person(BaseModel):
    name: str
    email: str

    @similarity_func("name", "email")
    def similarity_name_email(x: t.Any, y: t.Any) -> float:
        return 0.9 if x == y else 0.0

person_got = Person(name="John", email="john@example.com")
person_expected = Person(name="John", email="john@example.com")

result = person_got.compute_similarity(person_expected)
print(result.fields.name.value)  # 0.9
print(result.fields.email.value) # 0.9
```

## Built-in Similarity Functions

Cobjectric provides several built-in similarity functions in `cobjectric.similarities`:

### exact_similarity

Exact equality comparison (default):

```python
from cobjectric.similarities import exact_similarity

class Person(BaseModel):
    name: str = Spec(similarity_func=exact_similarity)
    age: int = Spec(similarity_func=exact_similarity)

person_got = Person(name="John", age=30)
person_expected = Person(name="John", age=30)

result = person_got.compute_similarity(person_expected)
print(result.fields.name.value)  # 1.0
print(result.fields.age.value)   # 1.0
```

### fuzzy_similarity_factory

Fuzzy string similarity using rapidfuzz:

```python
from cobjectric.similarities import fuzzy_similarity_factory

class Person(BaseModel):
    name: str = Spec(similarity_func=fuzzy_similarity_factory())

person_got = Person(name="John Doe")
person_expected = Person(name="john doe")

result = person_got.compute_similarity(person_expected)
print(result.fields.name.value)  # ~0.9 (high similarity despite case difference)
```

You can use different scorers:

```python
# Use different rapidfuzz scorers
fuzzy_ratio = fuzzy_similarity_factory(scorer="ratio")
fuzzy_partial = fuzzy_similarity_factory(scorer="partial_ratio")
fuzzy_token = fuzzy_similarity_factory(scorer="token_sort_ratio")
```

### numeric_similarity_factory

Numeric similarity with tolerance:

```python
from cobjectric.similarities import numeric_similarity_factory

class Person(BaseModel):
    age: int = Spec(similarity_func=numeric_similarity_factory())
    score: float = Spec(
        similarity_func=numeric_similarity_factory(max_difference=5.0)
    )

person_got = Person(age=30, score=10.0)
person_expected = Person(age=30, score=12.0)

result = person_got.compute_similarity(person_expected)
print(result.fields.age.value)   # 1.0 (exact match)
print(result.fields.score.value) # 0.6 (diff=2, 2/5=0.4, 1-0.4=0.6)
```

## Separate Weights for Similarity

Similarity uses its own weight system, separate from fill rate and fill rate accuracy weights:

```python
class Person(BaseModel):
    name: str = Spec(
        fill_rate_func=lambda x: 0.5,
        fill_rate_weight=2.0,  # Weight for compute_fill_rate()
        similarity_func=lambda got, exp: 1.0 if got == exp else 0.0,
        similarity_weight=1.5,  # Weight for compute_similarity()
    )

person_got = Person(name="John")
person_expected = Person(name="John")

# Fill rate result uses fill_rate_weight
fill_rate_result = person_got.compute_fill_rate()
print(fill_rate_result.fields.name.weight)  # 2.0

# Similarity result uses similarity_weight
similarity_result = person_got.compute_similarity(person_expected)
print(similarity_result.fields.name.weight)  # 1.5
```

## Similarity with Nested Models

Similarity works recursively with nested models:

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
    "name": "John",
    "address": {"street": "123 Main St", "city": "Anytown"},
})

result = person_got.compute_similarity(person_expected)

# Both have same name -> 1.0
print(result.fields.name.value)  # 1.0

# Both have same address values -> 1.0 for nested fields
print(result.fields.address.fields.street.value)  # 1.0
print(result.fields.address.fields.city.value)   # 1.0
```

## Similarity with List Fields

Similarity supports list fields with different behaviors depending on the list element type.

### List of Primitive Types

For `list[str]`, `list[int]`, etc., similarity compares the lists element by element:

```python
class Person(BaseModel):
    tags: list[str]

person_got = Person(tags=["python", "rust"])
person_expected = Person(tags=["python", "rust"])

result = person_got.compute_similarity(person_expected)
print(result.fields.tags.value)  # 1.0 (both have same lists)

person_got = Person(tags=["python", "rust"])
person_expected = Person(tags=["java", "go"])

result = person_got.compute_similarity(person_expected)
print(result.fields.tags.value)  # 0.0 (different lists)
```

### List of BaseModel

For `list[BaseModel]`, similarity compares items one by one. Only items that exist in both lists are compared:

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
        {"name": "Apple", "price": 1.0},
        {"name": "Banana", "price": 0.5},
    ],
})

result = order_got.compute_similarity(order_expected)

# Both have 2 items, all fields same -> similarity = 1.0 for all
print(len(result.fields.items))  # 2
print(result.fields.items[0].fields.name.value)   # 1.0
print(result.fields.items[0].fields.price.value)  # 1.0

# Aggregated access works too (recommended API)
print(result.fields.items.aggregated_fields.name.values)   # [1.0, 1.0]
print(result.fields.items.aggregated_fields.price.values)  # [1.0, 1.0]
```

**Different List Lengths**:

When lists have different lengths, only the items that exist in both lists are compared:

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

result = order_got.compute_similarity(order_expected)

# Only first item is compared (min(len(got), len(expected)))
print(len(result.fields.items))  # 1
print(result.fields.items[0].fields.name.value)  # 0.0 (different values)
```

## List Comparison Strategies

When comparing `list[BaseModel]` fields, you can use different comparison strategies to align items when list order may differ. By default, items are compared pairwise (by index).

For detailed information about all available strategies and how to use them, see [List Comparison Strategies](list_comparison.md).

The strategies work the same way for both similarity and fill rate accuracy computations.

## Duplicate Similarity Functions

A field can only have one `similarity_func`. If multiple functions are defined, `DuplicateSimilarityFuncError` is raised:

```python
from cobjectric import DuplicateSimilarityFuncError

class Person(BaseModel):
    name: str = Spec(similarity_func=lambda got, exp: 0.5)

    @similarity_func("name")
    def similarity_name(got: t.Any, expected: t.Any) -> float:
        return 0.6

person_got = Person(name="John")
person_expected = Person(name="Jane")

try:
    result = person_got.compute_similarity(person_expected)
except DuplicateSimilarityFuncError as e:
    print(f"Error: {e}")
```

## Related Topics

- [BaseModel](base_model.md) - Learn about the base model class
- [Field Types](field_types.md) - Learn about different field types
- [Nested Models](nested_models.md) - Learn about nested model structures
- [Fill Rate](fill_rate.md) - Learn about fill rate computation
- [List Comparison Strategies](list_comparison.md) - Detailed guide on list comparison strategies

