# Nested Models

You can define nested models by using another `BaseModel` subclass as a field type. When creating instances from dictionaries, nested dictionaries are automatically converted to model instances.

## Defining Nested Models

```python
from cobjectric import BaseModel

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
```

## Creating Nested Models from Dictionaries

When using `from_dict`, nested dictionaries are automatically converted to model instances:

```python
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
```

## Accessing Nested Model Fields

Nested models are accessed directly through the `.fields` attribute, not as `Field` instances:

```python
# Access the nested model
address = person.fields.address  # Returns an Address instance

# Access fields of the nested model
print(address.fields.street.value)  # "123 Main St"
print(address.fields.city.value)   # "Anytown"
```

## Creating Nested Models with Instances

You can also pass model instances directly when creating a model:

```python
address = Address(
    street="123 Main St",
    city="Anytown",
    state="CA",
    zip_code="12345",
    country="USA",
)

person = Person(
    name="John Doe",
    age=30,
    email="john.doe@example.com",
    is_active=True,
    address=address,
)
```

## Deeply Nested Models

You can nest models at multiple levels:

```python
class Country(BaseModel):
    name: str
    code: str

class Address(BaseModel):
    street: str
    city: str
    country: Country

class Person(BaseModel):
    name: str
    address: Address

person = Person.from_dict({
    "name": "John Doe",
    "address": {
        "street": "123 Main St",
        "city": "Anytown",
        "country": {
            "name": "United States",
            "code": "US",
        },
    },
})

# Access deeply nested fields
print(person.fields.address.fields.country.fields.name.value)  # "United States"
```

## Missing Nested Models

If a nested model is not provided or has an invalid type, it will have the value `MissingValue`:

```python
from cobjectric import MissingValue

person = Person.from_dict({
    "name": "John Doe",
    # address is missing
})

print(person.fields.address.value is MissingValue)  # True
```

## Related Topics

- [BaseModel](base_model.md) - Learn about the base model class
- [Field Types](field_types.md) - Learn about different field types
- [Path Access](path_access.md) - Learn about accessing nested fields by path notation

