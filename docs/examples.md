# Examples

This page contains practical examples demonstrating the various features of Cobjectric, including field handling, metrics computation, and advanced features.

## Basic Examples

### Example 1: Basic Model

```python
from cobjectric import BaseModel

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

product = Product(name="Laptop", price=999.99, in_stock=True)
print(product.fields.name.value)    # "Laptop"
print(product.fields.price.value)   # 999.99
```

### Example 2: Handling Missing Fields

```python
from cobjectric import MissingValue

class User(BaseModel):
    username: str
    email: str
    age: int

user = User(username="dave", email="dave@example.com")
# age is not provided
if user.fields.age.value is MissingValue:
    print("User age not provided")
```

### Example 3: Type Validation

```python
class User(BaseModel):
    username: str
    email: str
    age: int

# Wrong type for age
user = User(
    username="eve",
    email="eve@example.com",
    age="twenty",
)
print(user.fields.age.value is MissingValue)  # True
```

### Example 4: Creating from Dictionary

```python
class Person(BaseModel):
    name: str
    age: int
    email: str
    is_active: bool

# Create from dictionary
person = Person.from_dict({
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
    "is_active": True,
})

print(person.fields.name.value)  # "John Doe"
print(person.fields.age.value)   # 30
```

### Example 5: Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    email: str
    is_active: bool
    address: Address

# Create from dictionary with nested model
person = Person.from_dict({
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
    "is_active": True,
    "address": {
        "street": "123 Main St",
        "city": "Anytown",
        "state": "CA",
        "zip_code": "12345",
        "country": "USA",
    },
})

print(person.fields.name.value)  # "John Doe"
print(person.fields.address.fields.street.value)  # "123 Main St"
```

## Fill Rate Examples

### Example 6: Computing Fill Rate

```python
class UserProfile(BaseModel):
    username: str
    email: str
    bio: str
    profile_picture_url: str
    verified: bool

# Create a user with incomplete profile
user = UserProfile.from_dict({
    "username": "john_doe",
    "email": "john@example.com",
    "bio": "Software developer",
    # profile_picture_url is missing
    # verified is missing
})

result = user.compute_fill_rate()

print(result.fields.username.value)  # 1.0
print(result.fields.email.value)     # 1.0
print(result.fields.bio.value)       # 1.0
print(result.fields.profile_picture_url.value)  # 0.0 (missing)
print(result.fields.verified.value)  # 0.0 (missing)

# Get overall statistics
print(result.mean())   # 0.6 (3 out of 5 fields)
print(result.min())    # 0.0
print(result.max())    # 1.0
```

### Example 7: Custom Fill Rate Functions

```python
from cobjectric import BaseModel, Spec

class Article(BaseModel):
    title: str = Spec(fill_rate_func=lambda x: len(x) / 100)  # Based on length
    content: str = Spec(fill_rate_func=lambda x: len(x) / 1000)
    author: str

article = Article(
    title="My Article",  # 11 chars, 0.11 fill rate
    content="A" * 500,   # 500 chars, 0.5 fill rate
    author="John"        # 1.0 fill rate (default)
)

result = article.compute_fill_rate()
print(result.fields.title.value)    # 0.11
print(result.fields.content.value)  # 0.5
print(result.fields.author.value)   # 1.0
print(result.mean())  # 0.536...
```

### Example 8: Weighted Fill Rate

```python
from cobjectric import BaseModel, Spec

class DataSet(BaseModel):
    # Critical fields with higher weight
    required_field: str = Spec(weight=3.0)
    important_field: str = Spec(weight=2.0)
    # Optional field with lower weight
    nice_to_have: str = Spec(weight=1.0)

data = DataSet(
    required_field="Present",
    important_field="Present",
    nice_to_have="Missing"
)

result = data.compute_fill_rate()
# All fields are filled except nice_to_have
# But weighted mean: (1.0*3.0 + 1.0*2.0 + 0.0*1.0) / (3.0 + 2.0 + 1.0) = 5.0/6.0 = 0.833...
print(result.mean())  # 0.833...
```

### Example 9: Fill Rate with Nested Models

```python
class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

person = Person.from_dict({
    "name": "John",
    "address": {"street": "123 Main St"}  # city is missing
})

result = person.compute_fill_rate()
print(result.fields.name.value)  # 1.0

# Get nested results
address_result = result.fields.address
print(address_result.fields.street.value)  # 1.0
print(address_result.fields.city.value)    # 0.0
print(address_result.mean())  # 0.5
```

## Fill Rate Accuracy Examples

### Example 10: Comparing Data Completeness

```python
class Customer(BaseModel):
    name: str
    email: str
    phone: str

# Expected profile (what we want)
expected = Customer.from_dict({
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "555-1234"
})

# Actual data received
actual = Customer.from_dict({
    "name": "John Doe",
    "email": "john@example.com"
    # phone is missing
})

result = actual.compute_fill_rate_accuracy(expected)

print(result.fields.name.value)   # 1.0 (both have name)
print(result.fields.email.value)  # 1.0 (both have email)
print(result.fields.phone.value)  # 0.0 (expected has phone, actual doesn't)
print(result.mean())  # 0.666...
```

## Similarity Examples

### Example 11: Basic Similarity

```python
class Product(BaseModel):
    name: str
    price: float

# Reference product
reference = Product(name="Laptop", price=999.99)

# Product from vendor A
vendor_a = Product(name="Laptop", price=1099.99)

# Product from vendor B
vendor_b = Product(name="Desktop Computer", price=999.99)

# Compare vendor A
result_a = vendor_a.compute_similarity(reference)
print(result_a.fields.name.value)   # 1.0 (same name)
print(result_a.fields.price.value)  # 0.0 (different price)
print(result_a.mean())  # 0.5

# Compare vendor B
result_b = vendor_b.compute_similarity(reference)
print(result_b.fields.name.value)   # 0.0 (different name)
print(result_b.fields.price.value)  # 1.0 (same price)
print(result_b.mean())  # 0.5
```

### Example 12: Fuzzy String Similarity

```python
from cobjectric import BaseModel, Spec
from cobjectric.similarity import fuzzy_similarity_factory

class Person(BaseModel):
    name: str = Spec(similarity_func=fuzzy_similarity_factory())
    age: int

# Reference person
reference = Person(name="John Doe", age=30)

# Similar person with typo in name
actual = Person(name="Jon Doe", age=30)

result = actual.compute_similarity(reference)
print(result.fields.name.value)  # ~0.909... (very similar despite typo)
print(result.fields.age.value)   # 1.0 (exact match)
print(result.mean())  # ~0.954...
```

### Example 13: Numeric Similarity with Tolerance

```python
from cobjectric import BaseModel, Spec
from cobjectric.similarity import numeric_similarity_factory

class Measurement(BaseModel):
    temperature: float = Spec(
        similarity_func=numeric_similarity_factory(max_difference=5.0)
    )
    humidity: int = Spec(
        similarity_func=numeric_similarity_factory(max_difference=10)
    )

# Expected measurements
expected = Measurement(temperature=20.0, humidity=50)

# Actual measurements (slightly off)
actual = Measurement(temperature=22.5, humidity=58)

result = actual.compute_similarity(expected)
# Temperature: diff=2.5, tolerance=5.0, similarity = 1 - (2.5/5.0) = 0.5
print(result.fields.temperature.value)  # 0.5
# Humidity: diff=8, tolerance=10, similarity = 1 - (8/10) = 0.2
print(result.fields.humidity.value)  # 0.2
print(result.mean())  # 0.35
```

## Pre-defined Specs Examples

### Example 14: Using KeywordSpec

```python
from cobjectric import BaseModel
from cobjectric.specs import KeywordSpec

class Product(BaseModel):
    product_id: str = KeywordSpec()  # Converts int to str by default
    sku: str = KeywordSpec(strip=True, convert_int_to_str=True)

# ID comes as int from JSON
product = Product.from_dict({
    "product_id": 12345,  # int from JSON
    "sku": "  ABC-123  "
})

assert product.fields.product_id.value == "12345"  # Converted to string
assert product.fields.sku.value == "ABC-123"  # Stripped and converted
```

### Example 15: Using DatetimeSpec

```python
from datetime import timedelta

from cobjectric import BaseModel
from cobjectric.specs import DatetimeSpec

class Event(BaseModel):
    created_at: str = DatetimeSpec()  # Auto-detects format
    updated_at: str = DatetimeSpec(format="%Y-%m-%d %H:%M:%S")
    timestamp: str = DatetimeSpec(max_difference=timedelta(hours=1))

# Various input formats
event1 = Event(created_at="2024-01-15T10:30:00Z")
assert event1.fields.created_at.value == "2024-01-15T10:30:00Z"

event2 = Event(updated_at="2024-01-15 10:30:00")
assert event2.fields.updated_at.value == "2024-01-15T10:30:00"

# With timezone
event3 = Event(created_at="2024-01-15T10:30:00+05:00")
assert event3.fields.created_at.value == "2024-01-15T10:30:00+05:00"

# With different date format
event4 = Event(created_at="15/01/2024 10:30:00")
assert event4.fields.created_at.value == "2024-01-15T10:30:00"
```

### Example 16: Using NumericSpec with Type Coercion

```python
from cobjectric import BaseModel
from cobjectric.specs import NumericSpec

class Person(BaseModel):
    age: int = NumericSpec()  # Automatically coerces float to int
    score: float = NumericSpec(max_difference=0.5)

# JSON provides float for int field
person = Person.from_dict({
    "age": 30.0,  # float from JSON
    "score": 85.5
})

assert person.fields.age.value == 30
assert isinstance(person.fields.age.value, int)  # Coerced to int
assert person.fields.score.value == 85.5
```

## List Examples

### Example 17: List Fields

```python
class Team(BaseModel):
    name: str
    members: list[str]
    scores: list[int]

team = Team(
    name="Team A",
    members=["Alice", "Bob", "Charlie"],
    scores=[85, 90, 88]
)

print(team.fields.members.value)  # ["Alice", "Bob", "Charlie"]
print(team.fields.scores.value)   # [85, 90, 88]
```

### Example 15: List Comparison with Default Strategy (Pairwise)

```python
class Order(BaseModel):
    items: list['Item']

class Item(BaseModel):
    name: str
    quantity: int

# Reference order
reference = Order.from_dict({
    "items": [
        {"name": "Apple", "quantity": 5},
        {"name": "Banana", "quantity": 3},
    ]
})

# Actual order (same items, different quantities)
actual = Order.from_dict({
    "items": [
        {"name": "Apple", "quantity": 5},
        {"name": "Banana", "quantity": 4},
    ]
})

result = actual.compute_similarity(reference)
print(len(result.fields.items))  # 2 items compared

# First item
print(result.fields.items[0].fields.name.value)      # 1.0 (Apple == Apple)
print(result.fields.items[0].fields.quantity.value)  # 1.0 (5 == 5)

# Second item
print(result.fields.items[1].fields.name.value)      # 1.0 (Banana == Banana)
print(result.fields.items[1].fields.quantity.value)  # 0.0 (4 != 3)
```

### Example 16: List Comparison with Optimal Assignment Strategy

```python
from cobjectric import BaseModel, Spec, ListCompareStrategy

class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    items: list[Item] = Spec(
        list_compare_strategy=ListCompareStrategy.OPTIMAL_ASSIGNMENT
    )

# Reference order
reference = Order.from_dict({
    "items": [
        {"name": "Apple", "price": 1.0},
        {"name": "Banana", "price": 0.5},
    ]
})

# Actual order (items in different order)
actual = Order.from_dict({
    "items": [
        {"name": "Banana", "price": 0.5},
        {"name": "Apple", "price": 1.0},
    ]
})

result = actual.compute_similarity(reference)
# With optimal assignment, items are matched correctly despite different order
print(result.fields.items[0].fields.name.value)   # 1.0 (best match found)
print(result.fields.items[0].fields.price.value)  # 1.0
print(result.fields.items[1].fields.name.value)   # 1.0
print(result.fields.items[1].fields.price.value)  # 1.0
```

## Path Access Examples

### Example 17: Accessing Nested Fields by Path

```python
class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Person(BaseModel):
    name: str
    address: Address

person = Person.from_dict({
    "name": "John Doe",
    "address": {
        "street": "123 Main St",
        "city": "New York",
        "zip_code": "10001"
    }
})

# Access nested fields by path
print(person["name"].value)               # "John Doe"
print(person["address.city"].value)       # "New York"
print(person["address.zip_code"].value)   # "10001"
```

### Example 18: Path Access on Fill Rate Results

```python
class Company(BaseModel):
    name: str
    industry: str

class Office(BaseModel):
    address: str
    company: Company

office = Office.from_dict({
    "address": "456 Tech Ave",
    "company": {"name": "TechCorp"}  # industry is missing
})

result = office.compute_fill_rate()

# Access results by path
print(result["address"].value)               # 1.0
print(result["company.name"].value)          # 1.0
print(result["company.industry"].value)      # 0.0
```

### Example 19: List Index Path Access

```python
class Item(BaseModel):
    name: str
    price: float

class ShoppingCart(BaseModel):
    items: list[Item]

cart = ShoppingCart.from_dict({
    "items": [
        {"name": "Apple", "price": 1.0},
        {"name": "Banana", "price": 0.5},
    ]
})

# Access items by index using path notation
print(cart["items[0].name"].value)    # "Apple"
print(cart["items[1].price"].value)   # 0.5
```

## Field Specifications Examples

### Example 20: Field Normalizers and Spec

```python
from cobjectric import BaseModel, Spec, field_normalizer

class Article(BaseModel):
    title: str = Spec(
        metadata={"description": "Article title"},
        normalizer=lambda x: x.strip().title()
    )
    content: str
    tags: list[str]

    @field_normalizer("tags")
    def normalize_tags(tags):
        return [tag.lower().strip() for tag in tags]

article = Article(
    title="  hello world  ",
    content="Some content",
    tags=["  Python  ", "WEB   "]
)

print(article.fields.title.value)  # "Hello World"
print(article.fields.tags.value)   # ["python", "web"]
```

## Complex Real-World Example

### Example 21: Data Quality Assessment Pipeline

```python
from cobjectric import BaseModel, Spec, field_normalizer
from cobjectric.similarity import fuzzy_similarity_factory

class Contact(BaseModel):
    first_name: str = Spec(
        normalizer=lambda x: x.strip().title(),
        fill_rate_weight=2.0,  # Important field
    )
    last_name: str = Spec(
        normalizer=lambda x: x.strip().title(),
        fill_rate_weight=2.0,
    )
    email: str = Spec(
        normalizer=lambda x: x.strip().lower(),
        fill_rate_weight=1.5,
    )
    phone: str = Spec(fill_rate_weight=1.0)  # Optional

# Data entry with some issues
raw_data = Contact.from_dict({
    "first_name": "  john  ",
    "last_name": "  doe  ",
    "email": "  JOHN@EXAMPLE.COM  ",
    # phone is missing
})

# Assess data quality
fill_rate_result = raw_data.compute_fill_rate()
print(f"Data Completeness: {fill_rate_result.mean():.1%}")  # ~75%

# Compare with expected data
expected = Contact.from_dict({
    "first_name": "John",
    "last_name": "Doe",
    "email": "john@example.com",
    "phone": "555-1234"
})

accuracy_result = raw_data.compute_fill_rate_accuracy(expected)
print(f"Data Accuracy: {accuracy_result.mean():.1%}")  # ~75%

# Check similarity with fuzzy matching
similarity_result = raw_data.compute_similarity(expected)
print(f"Data Similarity: {similarity_result.mean():.1%}")  # High (normalized values match)
```

## Related Topics

- [Quick Start](quickstart.md) - Get started in 5 minutes
- [BaseModel](base_model.md) - Learn about the base model class
- [Field Types](field_types.md) - Learn about different field types
- [Nested Models](nested_models.md) - Learn about nested model structures
- [Fill Rate](fill_rate.md) - Learn about fill rate and fill rate accuracy
- [Similarity](similarity.md) - Learn about similarity computation
- [List Comparison Strategies](list_comparison.md) - Learn about list comparison strategies
- [Path Access](path_access.md) - Learn about accessing fields by path notation
- [Field Specifications](field_specs.md) - Learn about Spec() and field normalizers

## API Reference

See the [API Reference](reference.md) for the complete API documentation.
