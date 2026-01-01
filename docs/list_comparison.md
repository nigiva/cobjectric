# List Comparison Strategies

When comparing `list[BaseModel]` fields in similarity or fill rate accuracy computations, you can use different comparison strategies to align items when list order may differ.

## Available Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `pairwise` | Compare items by index (default) | Lists have same order |
| `levenshtein` | Align items preserving relative order | Lists with insertions/deletions |
| `optimal_assignment` | Find optimal one-to-one mapping | Lists with different order |

## Using list_compare_strategy

Set the strategy using `Spec(list_compare_strategy=...)`:

```python
from cobjectric import BaseModel, Spec, ListCompareStrategy

class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    # Default: pairwise comparison
    items_pairwise: list[Item]
    # Levenshtein alignment (preserves relative order)
    items_levenshtein: list[Item] = Spec(
        list_compare_strategy=ListCompareStrategy.LEVENSHTEIN
    )
    # Optimal assignment (Hungarian algorithm)
    items_optimal: list[Item] = Spec(
        list_compare_strategy=ListCompareStrategy.OPTIMAL_ASSIGNMENT
    )
```

You can also use strings:

```python
class Order(BaseModel):
    items: list[Item] = Spec(list_compare_strategy="levenshtein")
```

## Pairwise Strategy (Default)

Compares items by their index. Simple and fast, but requires lists to be in the same order:

```python
class Item(BaseModel):
    name: str

class Order(BaseModel):
    items: list[Item]  # Uses pairwise by default

order_got = Order.from_dict({"items": [{"name": "Apple"}, {"name": "Banana"}]})
order_expected = Order.from_dict({"items": [{"name": "Apple"}, {"name": "Banana"}]})

result = order_got.compute_similarity(order_expected)
print(result.fields.items[0].fields.name.value)  # 1.0 (Apple == Apple)
print(result.fields.items[1].fields.name.value)  # 1.0 (Banana == Banana)
```

If items are in different order, pairwise comparison will fail to match them:

```python
order_got = Order.from_dict({"items": [{"name": "Apple"}, {"name": "Banana"}]})
order_expected = Order.from_dict({"items": [{"name": "Banana"}, {"name": "Apple"}]})

result = order_got.compute_similarity(order_expected)
print(result.fields.items[0].fields.name.value)  # 0.0 (Apple != Banana)
print(result.fields.items[1].fields.name.value)  # 0.0 (Banana != Apple)
```

## Levenshtein Strategy

Uses dynamic programming to find the best alignment **while preserving relative order**. Good for lists with insertions or deletions:

```python
class Item(BaseModel):
    name: str

class Order(BaseModel):
    items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

# got: [Apple, Cherry, Banana]
# expected: [Apple, Banana]
# Best alignment: Apple-Apple, Banana-Banana (skip Cherry)
order_got = Order.from_dict({
    "items": [{"name": "Apple"}, {"name": "Cherry"}, {"name": "Banana"}]
})
order_expected = Order.from_dict({
    "items": [{"name": "Apple"}, {"name": "Banana"}]
})

result = order_got.compute_similarity(order_expected)
print(len(result.fields.items))  # 2
print(result.fields.items[0].fields.name.value)  # 1.0 (Apple)
print(result.fields.items[1].fields.name.value)  # 1.0 (Banana)
```

**Important**: Levenshtein **preserves relative order**. It cannot match items that would violate the original order:

```python
# got: [Apple, Banana]
# expected: [Banana, Apple]
# Levenshtein can only align ONE item (Apple-Apple OR Banana-Banana)
order_got = Order.from_dict({"items": [{"name": "Apple"}, {"name": "Banana"}]})
order_expected = Order.from_dict({"items": [{"name": "Banana"}, {"name": "Apple"}]})

result = order_got.compute_similarity(order_expected)
print(len(result.fields.items))  # 1 (only one item aligned)
```

## Optimal Assignment Strategy

Uses the Hungarian algorithm to find the **optimal one-to-one mapping** regardless of order. Best for lists where order doesn't matter:

```python
from cobjectric import BaseModel, Spec, ListCompareStrategy

class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    items: list[Item] = Spec(
        list_compare_strategy=ListCompareStrategy.OPTIMAL_ASSIGNMENT
    )

# got: [Apple, Banana]
# expected: [Banana, Apple]
# Optimal alignment: Apple-Apple, Banana-Banana
order_got = Order.from_dict({
    "items": [
        {"name": "Apple", "price": 1.0},
        {"name": "Banana", "price": 0.5},
    ]
})
order_expected = Order.from_dict({
    "items": [
        {"name": "Banana", "price": 0.5},
        {"name": "Apple", "price": 1.0},
    ]
})

result = order_got.compute_similarity(order_expected)
print(len(result.fields.items))  # 2
# All items are perfectly matched
print(result.fields.items[0].fields.name.value)   # 1.0
print(result.fields.items[0].fields.price.value)  # 1.0
print(result.fields.items[1].fields.name.value)   # 1.0
print(result.fields.items[1].fields.price.value)  # 1.0
```

**Note**: The `optimal_assignment` strategy requires `scipy` to be installed:

```bash
pip install scipy
```

## Strategy Comparison

| Scenario | Pairwise | Levenshtein | Optimal Assignment |
|----------|----------|-------------|-------------------|
| Same order | ✅ Best | ✅ Works | ✅ Works |
| Insertions/deletions | ❌ Poor | ✅ Best | ✅ Works |
| Different order | ❌ Poor | ❌ Poor | ✅ Best |
| Performance | ⚡ O(n) | 📊 O(n×m) | 📊 O(n³) |

## Usage with Similarity

List comparison strategies are used when computing similarity between two models. See [Similarity](similarity.md) for details.

## Usage with Fill Rate Accuracy

List comparison strategies are also used when computing fill rate accuracy between two models. See [Fill Rate Accuracy](fill_rate.md#fill-rate-accuracy-with-list-fields) for details.

## Invalid Usage

Using `list_compare_strategy` on non-`list[BaseModel]` fields raises `InvalidListCompareStrategyError`:

```python
from cobjectric import InvalidListCompareStrategyError

# Error: Using on a non-list field
class Person(BaseModel):
    name: str = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

person_got = Person(name="John")
person_expected = Person(name="Jane")

try:
    person_got.compute_similarity(person_expected)
except InvalidListCompareStrategyError as e:
    print(f"Error: {e}")

# Error: Using on list[Primitive] (only list[BaseModel] is supported)
class Person(BaseModel):
    tags: list[str] = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

person_got = Person(tags=["python"])
person_expected = Person(tags=["rust"])

try:
    person_got.compute_similarity(person_expected)
except InvalidListCompareStrategyError as e:
    print(f"Error: {e}")
```

## Related Topics

- [Similarity](similarity.md) - Learn about similarity computation
- [Fill Rate](fill_rate.md) - Learn about fill rate and fill rate accuracy computation

