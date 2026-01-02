# Pandas Export

Cobjectric provides optional pandas integration for exporting results to pandas Series and DataFrames. This is useful for statistical analysis and data visualization.

## Installation

To use pandas export features, install cobjectric with the pandas extra:

```bash
pip install cobjectric[pandas]
```

## Exporting to Series

You can export a single `ModelResult` to a pandas Series using `to_series()`:

```python
from cobjectric import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str

person = Person(name="John", age=30, email="john@example.com")
result = person.compute_fill_rate()
series = result.to_series()

print(series)
# name     1.0
# age      1.0
# email    1.0
# dtype: float64
```

### Nested Models

For nested models, field paths use dot notation:

```python
class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

person = Person(
    name="John",
    address=Address(street="123 Main St", city="Anytown"),
)
result = person.compute_fill_rate()
series = result.to_series()

print(series)
# name           1.0
# address.street 1.0
# address.city   1.0
# dtype: float64
```

### List Fields

For `list[BaseModel]` fields, values are aggregated using `mean()` across all items:

```python
class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    items: list[Item]

order = Order(
    items=[
        Item(name="Apple", price=1.0),
        Item(name="Banana", price=0.5),
    ]
)
result = order.compute_fill_rate()
series = result.to_series()

print(series)
# items.name  1.0
# items.price 1.0
# dtype: float64
```

## Combining Results

You can combine multiple `ModelResult` instances using the `+` operator to create a `ModelResultCollection`:

```python
person1 = Person(name="John", age=30, email="john@example.com")
person2 = Person(name="Jane", age=25)
person3 = Person(name="Bob", age=40, email="bob@example.com")

result1 = person1.compute_fill_rate()
result2 = person2.compute_fill_rate()
result3 = person3.compute_fill_rate()

collection = result1 + result2 + result3
```

### Incompatible Results

All results in a collection must come from the same model type. If you try to combine results from different models, `IncompatibleModelResultError` is raised:

```python
class Person(BaseModel):
    name: str

class Product(BaseModel):
    title: str

person = Person(name="John")
product = Product(title="Book")

result1 = person.compute_fill_rate()
result2 = product.compute_fill_rate()

try:
    collection = result1 + result2
except IncompatibleModelResultError as e:
    print(f"Error: {e}")
```

## Exporting to DataFrame

You can export a `ModelResultCollection` to a pandas DataFrame using `to_dataframe()`:

```python
person1 = Person(name="John", age=30, email="john@example.com")
person2 = Person(name="Jane", age=25)
person3 = Person(name="Bob", age=40, email="bob@example.com")

result1 = person1.compute_fill_rate()
result2 = person2.compute_fill_rate()
result3 = person3.compute_fill_rate()

collection = result1 + result2 + result3
df = collection.to_dataframe()

print(df)
#    name  age  email
# 0   1.0  1.0    1.0
# 1   1.0  1.0    0.0
# 2   1.0  1.0    1.0
```

Each row represents one `ModelResult`, and columns are field paths (using dot notation for nested models).

## Statistical Methods

`ModelResultCollection` provides statistical methods for aggregating results:

### mean()

Calculate mean value for each field:

```python
collection = result1 + result2 + result3
means = collection.mean()

print(means)
# {'name': 1.0, 'age': 1.0, 'email': 0.6666666666666666}
```

### std()

Calculate standard deviation for each field:

```python
stds = collection.std()
print(stds)
# {'name': 0.0, 'age': 0.0, 'email': 0.4714045207910317}
```

### min() and max()

Get minimum and maximum values:

```python
mins = collection.min()
maxs = collection.max()
```

### quantile()

Calculate quantiles:

```python
median = collection.quantile(0.5)
q75 = collection.quantile(0.75)
```

## Working with Different Metrics

The pandas export works with all three metric types:

```python
# Fill rate
fill_rate_result = person.compute_fill_rate()
series = fill_rate_result.to_series()

# Fill rate accuracy
accuracy_result = person.compute_fill_rate_accuracy(expected_person)
series = accuracy_result.to_series()

# Similarity
similarity_result = person.compute_similarity(expected_person)
series = similarity_result.to_series()
```

All three return `ModelResult` instances, so they all support `to_series()` and can be combined into collections.

## Error Handling

If pandas is not installed, `to_series()` and `to_dataframe()` will raise `ImportError`:

```python
try:
    series = result.to_series()
except ImportError as e:
    print(f"Error: {e}")
    # Error: pandas is required for to_series() and to_dataframe().
    # Install it with: pip install cobjectric[pandas]
```

## Related Topics

- [BaseModel](base_model.md) - Learn about the base model class
- [Fill Rate](fill_rate.md) - Learn about fill rate computation
- [Fill Rate Accuracy](fill_rate_accuracy.md) - Learn about fill rate accuracy
- [Similarity](similarity.md) - Learn about similarity computation

