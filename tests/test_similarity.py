import typing as t

import pytest

from cobjectric import (
    AggregatedFieldResult,
    BaseModel,
    DuplicateSimilarityFuncError,
    InvalidFillRateValueError,
    InvalidWeightError,
    ListResult,
    MissingValue,
    ModelResult,
    Spec,
    similarity_func,
)
from cobjectric.similarities import (
    exact_similarity,
    fuzzy_similarity_factory,
    numeric_similarity_factory,
)


def test_default_similarity_both_same_value() -> None:
    """Test that default similarity returns 1.0 when values are equal."""

    class Person(BaseModel):
        name: str
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person(name="John", age=30)

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.value == 1.0
    assert result.fields.age.value == 1.0


def test_default_similarity_both_different_value() -> None:
    """Test that default similarity returns 0.0 when values differ."""

    class Person(BaseModel):
        name: str
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person(name="Jane", age=25)

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.value == 0.0
    assert result.fields.age.value == 0.0


def test_default_similarity_both_missing() -> None:
    """Test that default similarity returns 1.0 when both are MissingValue."""

    class Person(BaseModel):
        name: str
        age: int

    person_got = Person()
    person_expected = Person()

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.value == 1.0
    assert result.fields.age.value == 1.0


def test_default_similarity_one_missing() -> None:
    """Test that default similarity returns 0.0 when one is MissingValue."""

    class Person(BaseModel):
        name: str
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person()

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.value == 0.0
    assert result.fields.age.value == 0.0


def test_default_similarity_mixed() -> None:
    """Test that default similarity works correctly with mixed cases."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person_got = Person(name="John", age=30)
    person_expected = Person(name="John", email="john@example.com")

    result = person_got.compute_similarity(person_expected)

    # name: both same -> 1.0
    assert result.fields.name.value == 1.0
    # age: got filled, expected missing -> 0.0
    assert result.fields.age.value == 0.0
    # email: got missing, expected filled -> 0.0
    assert result.fields.email.value == 0.0


def test_spec_similarity_func() -> None:
    """Test that Spec(similarity_func=...) works correctly."""

    def custom_similarity(got: t.Any, expected: t.Any) -> float:
        if got is MissingValue and expected is MissingValue:
            return 1.0
        if got is not MissingValue and expected is not MissingValue:
            return 0.8 if got == expected else 0.3
        return 0.0

    class Person(BaseModel):
        name: str = Spec(similarity_func=custom_similarity)
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person(name="John", age=25)

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.value == 0.8  # Custom func: same value
    assert result.fields.age.value == 0.0  # Default func: different value


def test_spec_similarity_weight() -> None:
    """Test that Spec(similarity_weight=...) works correctly."""

    class Person(BaseModel):
        name: str = Spec(similarity_weight=2.0)
        age: int = Spec(similarity_weight=0.5)

    person_got = Person(name="John", age=30)
    person_expected = Person(name="John", age=30)

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.weight == 2.0
    assert result.fields.age.weight == 0.5


def test_decorator_similarity_func() -> None:
    """Test that @similarity_func decorator works."""

    class Person(BaseModel):
        name: str
        age: int

        @similarity_func("name")
        def similarity_name(got: t.Any, expected: t.Any) -> float:
            return 0.7 if got == expected else 0.0

    person_got = Person(name="John", age=30)
    person_expected = Person(name="John", age=25)

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.value == 0.7
    assert result.fields.age.value == 0.0  # Default func: different value


def test_decorator_similarity_weight() -> None:
    """Test that @similarity_func with weight works."""

    class Person(BaseModel):
        name: str
        age: int

        @similarity_func("name", weight=2.0)
        def similarity_name(got: t.Any, expected: t.Any) -> float:
            return 1.0 if got == expected else 0.0

    person_got = Person(name="John", age=30)
    person_expected = Person(name="John", age=30)

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.weight == 2.0
    assert result.fields.age.weight == 1.0


def test_decorator_similarity_multiple_fields() -> None:
    """Test that @similarity_func works with multiple fields."""

    class Person(BaseModel):
        name: str
        email: str
        age: int

        @similarity_func("name", "email")
        def similarity_name_email(got: t.Any, expected: t.Any) -> float:
            return 0.9 if got == expected else 0.0

    person_got = Person(name="John", email="john@example.com", age=30)
    person_expected = Person(name="John", email="john@example.com", age=25)

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.value == 0.9
    assert result.fields.email.value == 0.9
    assert result.fields.age.value == 0.0  # Default func


def test_decorator_similarity_pattern() -> None:
    """Test that @similarity_func supports glob patterns."""

    class Person(BaseModel):
        name_first: str
        name_last: str
        age: int

        @similarity_func("name_*")
        def similarity_name_fields(got: t.Any, expected: t.Any) -> float:
            return 0.8 if got == expected else 0.0

    person_got = Person(name_first="John", name_last="Doe", age=30)
    person_expected = Person(name_first="John", name_last="Doe", age=25)

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name_first.value == 0.8
    assert result.fields.name_last.value == 0.8
    assert result.fields.age.value == 0.0  # Default func


def test_weights_are_independent() -> None:
    """Test that similarity_weight is independent from other weights."""

    class Person(BaseModel):
        name: str = Spec(
            fill_rate_func=lambda x: 0.5,
            fill_rate_weight=2.0,
            similarity_weight=1.5,
        )
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person(name="John", age=30)

    # Fill rate result
    fill_rate_result = person_got.compute_fill_rate()
    assert fill_rate_result.fields.name.weight == 2.0

    # Similarity result
    similarity_result = person_got.compute_similarity(person_expected)
    assert similarity_result.fields.name.weight == 1.5


def test_decorator_weight_overrides_spec() -> None:
    """Test that decorator weight overrides Spec similarity_weight."""

    class Person(BaseModel):
        name: str = Spec(similarity_weight=1.0)

        @similarity_func("name", weight=2.0)
        def similarity_name(got: t.Any, expected: t.Any) -> float:
            return 1.0 if got == expected else 0.0

    person_got = Person(name="John")
    person_expected = Person(name="John")

    result = person_got.compute_similarity(person_expected)

    # Decorator weight (2.0) should override Spec weight (1.0)
    assert result.fields.name.weight == 2.0


def test_duplicate_similarity_func_raises() -> None:
    """Test that Spec + decorator on same field raises DuplicateSimilarityFuncError."""

    class Person(BaseModel):
        name: str = Spec(similarity_func=lambda got, exp: 0.5)

        @similarity_func("name")
        def similarity_name(got: t.Any, expected: t.Any) -> float:
            return 0.6

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    with pytest.raises(DuplicateSimilarityFuncError):
        person_got.compute_similarity(person_expected)


def test_duplicate_similarity_func_decorators_raises() -> None:
    """Test that multiple decorators on same field raises DuplicateSimilarityFuncError."""

    class Person(BaseModel):
        name: str

        @similarity_func("name")
        def similarity_name1(got: t.Any, expected: t.Any) -> float:
            return 0.5

        @similarity_func("name")
        def similarity_name2(got: t.Any, expected: t.Any) -> float:
            return 0.6

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    with pytest.raises(DuplicateSimilarityFuncError):
        person_got.compute_similarity(person_expected)


def test_similarity_weight_negative_in_spec_raises() -> None:
    """Test that negative similarity_weight in Spec() raises InvalidWeightError."""

    with pytest.raises(InvalidWeightError) as exc_info:
        Spec(similarity_weight=-1.0)

    assert "Invalid similarity_weight in Spec" in str(exc_info.value)
    assert "-1.0" in str(exc_info.value)


def test_similarity_weight_negative_in_decorator_raises() -> None:
    """Test that negative weight in decorator raises InvalidWeightError."""

    with pytest.raises(InvalidWeightError) as exc_info:

        class Person(BaseModel):
            name: str

            @similarity_func("name", weight=-1.0)
            def similarity_name(got: t.Any, expected: t.Any) -> float:
                return 1.0

    assert "Invalid similarity_weight in decorator" in str(exc_info.value)
    assert "-1.0" in str(exc_info.value)


def test_similarity_weight_zero_allowed() -> None:
    """Test that similarity_weight = 0.0 is allowed."""

    class Person(BaseModel):
        name: str = Spec(similarity_weight=0.0)
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person(name="John", age=30)

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.weight == 0.0
    assert result.fields.age.weight == 1.0


def test_similarity_nested_models() -> None:
    """Test that similarity works with nested models."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person.from_dict(
        {
            "name": "John",
            "address": {"street": "123 Main St", "city": "Anytown"},
        }
    )
    person_expected = Person.from_dict(
        {
            "name": "John",
            "address": {"street": "123 Main St", "city": "Anytown"},
        }
    )

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.value == 1.0
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 1.0
    assert result.fields.address.fields.city.value == 1.0


def test_similarity_nested_different_values() -> None:
    """Test that similarity works when nested model values differ."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person.from_dict(
        {
            "name": "John",
            "address": {"street": "123 Main St", "city": "Anytown"},
        }
    )
    person_expected = Person.from_dict(
        {
            "name": "John",
            "address": {"street": "456 Oak Ave", "city": "Somewhere"},
        }
    )

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.value == 1.0
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_similarity_nested_missing() -> None:
    """Test that similarity works when nested model is missing."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person(name="John")
    person_expected = Person.from_dict(
        {
            "name": "John",
            "address": {"street": "456 Oak Ave", "city": "Somewhere"},
        }
    )

    result = person_got.compute_similarity(person_expected)

    assert result.fields.name.value == 1.0
    assert isinstance(result.fields.address, ModelResult)
    # got missing, expected filled -> 0.0
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_similarity_nested_both_missing() -> None:
    """Test similarity when nested model type field is MissingValue in both."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    result = person_got.compute_similarity(person_expected)

    # Both missing -> similarity = 1.0 for all nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 1.0
    assert result.fields.address.fields.city.value == 1.0


def test_similarity_list_primitive_both_same() -> None:
    """Test similarity for list[Primitive] when both have same values."""

    class Person(BaseModel):
        tags: list[str]

    person_got = Person(tags=["python", "rust"])
    person_expected = Person(tags=["python", "rust"])

    result = person_got.compute_similarity(person_expected)

    # Both have same list -> similarity = 1.0
    assert result.fields.tags.value == 1.0


def test_similarity_list_primitive_different() -> None:
    """Test similarity for list[Primitive] when values differ."""

    class Person(BaseModel):
        tags: list[str]

    person_got = Person(tags=["python", "rust"])
    person_expected = Person(tags=["java", "go"])

    result = person_got.compute_similarity(person_expected)

    # Different lists -> similarity = 0.0
    assert result.fields.tags.value == 0.0


def test_similarity_list_primitive_one_missing() -> None:
    """Test similarity for list[Primitive] when one is missing."""

    class Person(BaseModel):
        tags: list[str]

    person_got = Person(tags=["python"])
    person_expected = Person()

    result = person_got.compute_similarity(person_expected)

    # got has list, expected is MissingValue -> similarity = 0.0
    assert result.fields.tags.value == 0.0


def test_similarity_list_primitive_both_empty() -> None:
    """Test similarity for list[Primitive] when both are empty."""

    class Person(BaseModel):
        tags: list[str]

    person_got = Person(tags=[])
    person_expected = Person(tags=[])

    result = person_got.compute_similarity(person_expected)

    # Both have empty lists -> similarity = 1.0
    assert result.fields.tags.value == 1.0


def test_similarity_list_basemodel_same_items() -> None:
    """Test similarity for list[BaseModel] with same items."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana", "price": 0.5},
            ],
        }
    )
    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana", "price": 0.5},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    # Both have 2 items, all fields same -> similarity = 1.0 for all
    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0


def test_similarity_list_basemodel_different_items() -> None:
    """Test similarity for list[BaseModel] with different items."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana", "price": 0.5},
            ],
        }
    )
    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "Orange", "price": 2.0},
                {"name": "Cherry", "price": 3.0},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    # Both have 2 items, all fields different -> similarity = 0.0 for all
    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 0.0
    assert result.fields.items[0].fields.price.value == 0.0


def test_similarity_list_basemodel_different_count() -> None:
    """Test similarity for list[BaseModel] with different item counts."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
            ],
        }
    )
    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "Orange"},
                {"name": "Cherry"},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    # got has 1 item, expected has 2 items
    # We compare item by item, so only first item is compared
    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 1
    assert result.fields.items[0].fields.name.value == 0.0


def test_similarity_list_basemodel_aggregated() -> None:
    """Test that aggregated access works for similarity results."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana", "price": 0.5},
            ],
        }
    )
    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Cherry", "price": 3.0},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    # name: both items have same name -> [1.0, 0.0]
    # price: item 0 same, item 1 different -> [1.0, 0.0]
    assert isinstance(result.fields.items.aggregated_fields.name, AggregatedFieldResult)
    assert result.fields.items.aggregated_fields.name.values == [1.0, 0.0]
    assert result.fields.items.aggregated_fields.price.values == [1.0, 0.0]


def test_similarity_mean_with_weights() -> None:
    """Test that mean() uses similarity_weight for weighted mean."""

    class Person(BaseModel):
        name: str = Spec(similarity_weight=2.0)
        age: int = Spec(similarity_weight=1.0)
        email: str = Spec(similarity_weight=1.0)

    person_got = Person(name="John", age=30, email="john@example.com")
    person_expected = Person(name="John", age=30, email="john@example.com")

    result = person_got.compute_similarity(person_expected)

    # All same -> 1.0, weights: 2.0, 1.0, 1.0
    # mean = (1.0 * 2.0 + 1.0 * 1.0 + 1.0 * 1.0) / (2.0 + 1.0 + 1.0) = 4.0 / 4.0 = 1.0
    assert result.mean() == pytest.approx(1.0)


def test_similarity_mean_mixed() -> None:
    """Test mean() with mixed similarity values."""

    class Person(BaseModel):
        name: str = Spec(similarity_weight=2.0)
        age: int = Spec(similarity_weight=1.0)
        email: str = Spec(similarity_weight=1.0)

    person_got = Person(name="John", age=30, email="john@example.com")
    person_expected = Person(name="John", age=25, email="jane@example.com")

    result = person_got.compute_similarity(person_expected)

    # name: same -> 1.0, weight=2.0
    # age: different -> 0.0, weight=1.0
    # email: different -> 0.0, weight=1.0
    # mean = (1.0 * 2.0 + 0.0 * 1.0 + 0.0 * 1.0) / (2.0 + 1.0 + 1.0) = 2.0 / 4.0 = 0.5
    assert result.mean() == pytest.approx(0.5)


def test_similarity_invalid_value_raises() -> None:
    """Test that invalid similarity value raises InvalidFillRateValueError."""

    class Person(BaseModel):
        name: str = Spec(similarity_func=lambda got, exp: 1.5)

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    with pytest.raises(InvalidFillRateValueError):
        person_got.compute_similarity(person_expected)


def test_similarity_empty_model() -> None:
    """Test that empty model works correctly."""

    class EmptyModel(BaseModel):
        pass

    instance_got = EmptyModel()
    instance_expected = EmptyModel()

    result = instance_got.compute_similarity(instance_expected)

    assert isinstance(result, ModelResult)
    assert result.mean() == 0.0


def test_exact_similarity_equal() -> None:
    """Test exact_similarity with equal values."""

    assert exact_similarity("John", "John") == 1.0
    assert exact_similarity(10, 10) == 1.0
    assert exact_similarity(3.14, 3.14) == 1.0
    assert exact_similarity(True, True) == 1.0
    assert exact_similarity(None, None) == 1.0


def test_exact_similarity_different() -> None:
    """Test exact_similarity with different values."""

    assert exact_similarity("John", "Jane") == 0.0
    assert exact_similarity(10, 11) == 0.0
    assert exact_similarity(3.14, 3.15) == 0.0
    assert exact_similarity(True, False) == 0.0
    assert exact_similarity("John", None) == 0.0


def test_fuzzy_similarity_factory_exact_match() -> None:
    """Test fuzzy_similarity_factory with exact match."""

    fuzzy = fuzzy_similarity_factory()
    assert fuzzy("John Doe", "John Doe") == pytest.approx(1.0, abs=0.01)


def test_fuzzy_similarity_factory_partial() -> None:
    """Test fuzzy_similarity_factory with partial match."""

    fuzzy = fuzzy_similarity_factory()
    result = fuzzy("John Doe", "John")
    assert 0.0 < result < 1.0


def test_fuzzy_similarity_factory_different() -> None:
    """Test fuzzy_similarity_factory with different strings."""

    fuzzy = fuzzy_similarity_factory()
    result = fuzzy("John Doe", "Jane Smith")
    assert 0.0 < result < 1.0


def test_fuzzy_similarity_factory_case_insensitive() -> None:
    """Test fuzzy_similarity_factory with case differences."""

    fuzzy = fuzzy_similarity_factory()
    result = fuzzy("John Doe", "john doe")
    assert (
        result > 0.7
    )  # Should be high similarity (ratio is case-sensitive but still high)


def test_fuzzy_similarity_factory_none() -> None:
    """Test fuzzy_similarity_factory with None values."""

    fuzzy = fuzzy_similarity_factory()
    assert fuzzy("John", None) == 0.0
    assert fuzzy(None, "John") == 0.0
    assert fuzzy(None, None) == 0.0


def test_fuzzy_similarity_factory_different_scorers() -> None:
    """Test fuzzy_similarity_factory with different scorer types."""

    ratio = fuzzy_similarity_factory(scorer="ratio")
    partial = fuzzy_similarity_factory(scorer="partial_ratio")
    token_sort = fuzzy_similarity_factory(scorer="token_sort_ratio")

    result1 = ratio("John Doe", "Doe John")
    result2 = partial("John Doe", "Doe John")
    result3 = token_sort("John Doe", "Doe John")

    # All should return valid similarity values
    assert 0.0 <= result1 <= 1.0
    assert 0.0 <= result2 <= 1.0
    assert 0.0 <= result3 <= 1.0


def test_numeric_similarity_factory_exact_mode_equal() -> None:
    """Test numeric_similarity_factory in exact mode with equal values."""

    exact = numeric_similarity_factory()
    assert exact(10, 10) == 1.0
    assert exact(10.5, 10.5) == 1.0
    assert exact(0, 0) == 1.0


def test_numeric_similarity_factory_exact_mode_different() -> None:
    """Test numeric_similarity_factory in exact mode with different values."""

    exact = numeric_similarity_factory()
    assert exact(10, 11) == 0.0
    assert exact(10.5, 10.6) == 0.0
    assert exact(0, 1) == 0.0


def test_numeric_similarity_factory_gradual_mode() -> None:
    """Test numeric_similarity_factory in gradual mode."""

    gradual = numeric_similarity_factory(max_difference=5.0)
    assert gradual(10, 10) == pytest.approx(1.0)
    assert gradual(10, 12) == pytest.approx(0.6)  # diff=2, 2/5=0.4, 1-0.4=0.6
    assert gradual(10, 15) == pytest.approx(0.0)  # diff=5, 5/5=1.0, 1-1.0=0.0
    assert gradual(10, 20) == pytest.approx(0.0)  # diff>max, capped at 0


def test_numeric_similarity_factory_gradual_mode_negative() -> None:
    """Test numeric_similarity_factory in gradual mode with negative values."""

    gradual = numeric_similarity_factory(max_difference=5.0)
    assert gradual(-10, -10) == pytest.approx(1.0)
    assert gradual(-10, -12) == pytest.approx(0.6)
    assert gradual(-10, -15) == pytest.approx(0.0)


def test_numeric_similarity_factory_gradual_mode_mixed() -> None:
    """Test numeric_similarity_factory in gradual mode with mixed signs."""

    gradual = numeric_similarity_factory(max_difference=10.0)
    assert gradual(5, -5) == pytest.approx(0.0)  # diff=10, 10/10=1.0, 1-1.0=0.0
    assert gradual(5, 0) == pytest.approx(0.5)  # diff=5, 5/10=0.5, 1-0.5=0.5


def test_numeric_similarity_factory_none() -> None:
    """Test numeric_similarity_factory with None values."""

    exact = numeric_similarity_factory()
    gradual = numeric_similarity_factory(max_difference=5.0)

    assert exact(10, None) == 0.0
    assert exact(None, 10) == 0.0
    assert exact(None, None) == 0.0
    assert gradual(10, None) == 0.0
    assert gradual(None, 10) == 0.0


def test_numeric_similarity_factory_invalid_conversion() -> None:
    """Test numeric_similarity_factory with non-numeric values."""

    exact = numeric_similarity_factory()
    gradual = numeric_similarity_factory(max_difference=5.0)

    assert exact("not a number", 10) == 0.0
    assert gradual(10, "not a number") == 0.0
    assert exact("abc", "def") == 0.0


def test_numeric_similarity_factory_invalid_max_difference() -> None:
    """Test numeric_similarity_factory raises ValueError for invalid max_difference."""

    with pytest.raises(ValueError, match="max_difference must be > 0"):
        numeric_similarity_factory(max_difference=0.0)

    with pytest.raises(ValueError, match="max_difference must be > 0"):
        numeric_similarity_factory(max_difference=-1.0)


def test_numeric_similarity_factory_string_numbers() -> None:
    """Test numeric_similarity_factory with string numbers."""

    exact = numeric_similarity_factory()
    gradual = numeric_similarity_factory(max_difference=5.0)

    assert exact("10", 10) == 1.0
    assert exact(10, "10") == 1.0
    assert gradual("10", "12") == pytest.approx(0.6)


def test_similarity_expected_field_not_in_expected() -> None:
    """Test similarity when expected field doesn't exist in expected model."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person_got = Person(name="John", age=30, email="john@example.com")
    person_expected = Person(name="Jane", age=25)

    # Manually remove 'email' from expected._fields to simulate missing field
    del person_expected._fields["email"]

    result = person_got.compute_similarity(person_expected)

    # email exists in got but NOT in expected._fields -> expected_value = MissingValue
    # got.email has value, expected.email is MissingValue -> similarity = 0.0
    assert result.fields.email.value == 0.0


def test_similarity_nested_expected_missing() -> None:
    """Test similarity when nested model is present in got but missing in expected."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person.from_dict(
        {
            "name": "John",
            "address": {"street": "123 Main St", "city": "Anytown"},
        }
    )
    person_expected = Person(name="Jane")

    result = person_got.compute_similarity(person_expected)

    # got has address, expected doesn't -> similarity = 0.0 for nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_similarity_list_basemodel_type_mismatch() -> None:
    """Test similarity when list item is not BaseModel instance."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
            ],
        }
    )
    # Manually set an item to a non-BaseModel value
    order_got._fields["items"].value[0] = "not a model"  # type: ignore

    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "Orange"},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    # Type mismatch should create empty result
    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 1
    # All fields should be 0.0 due to type mismatch
    assert result.fields.items[0].fields.name.value == 0.0


def test_similarity_nested_with_internal_fields() -> None:
    """Test similarity when nested model has internal fields with underscore."""

    class Address(BaseModel):
        street: str
        _internal: str = None  # Internal field

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    result = person_got.compute_similarity(person_expected)

    # Both address missing -> similarity = 1.0 for all fields
    # Internal fields (_internal) should be skipped
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 1.0


def test_similarity_nested_expected_not_basemodel() -> None:
    """Test similarity when nested field type is Address but expected is not BaseModel."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person.from_dict(
        {"name": "John", "address": {"street": "123 Main St", "city": "Anytown"}}
    )
    person_expected = Person(name="Jane")
    # Manually set address to a non-BaseModel value to test the edge case
    from cobjectric.field import Field
    from cobjectric.field_spec import FieldSpec

    person_expected._fields["address"] = Field(
        name="address", type=Address, value="not a model", spec=FieldSpec()
    )

    result = person_got.compute_similarity(person_expected)

    # Expected is not BaseModel -> similarity = 0.0 for all nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_similarity_nested_expected_missing_with_internal_fields() -> None:
    """Test similarity when nested model expected is missing and has internal fields."""

    class Address(BaseModel):
        street: str
        _internal: str = None  # Internal field

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person.from_dict(
        {"name": "John", "address": {"street": "123 Main St"}}
    )
    person_expected = Person(name="Jane")

    result = person_got.compute_similarity(person_expected)

    # got has address, expected doesn't -> similarity = 0.0 for nested fields
    # Internal fields (_internal) should be skipped
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0


def test_similarity_list_basemodel_type_mismatch_with_internal_fields() -> None:
    """Test similarity when list item type has private fields and type mismatch."""

    class Item(BaseModel):
        name: str
        _private: str  # Private field

    class Order(BaseModel):
        items: list[Item]

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
            ],
        }
    )
    # Manually set an item to a non-BaseModel value
    order_got._fields["items"].value[0] = "not a model"  # type: ignore

    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "Orange"},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    # Type mismatch should create empty result, private field should be skipped
    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 1
    # Only name field should be present (private field skipped)
    assert "name" in result.fields.items[0]._fields
    assert "_private" not in result.fields.items[0]._fields


def test_similarity_nested_field_expected_not_basemodel_instance() -> None:
    """Test similarity when nested Field has BaseModel value but expected is not BaseModel instance."""

    class Address(BaseModel):
        street: str
        city: str
        _internal: str = None

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person.from_dict(
        {"name": "John", "address": {"street": "123 Main", "city": "NYC"}}
    )
    person_expected = Person(name="Jane")

    # Set address in expected to a non-BaseModel value (string)
    # This simulates the case where expected_value is not a BaseModel
    from cobjectric.field import Field
    from cobjectric.field_spec import FieldSpec

    person_expected._fields["address"] = Field(
        name="address",
        type=Address,
        value="string_not_basemodel",
        spec=FieldSpec(),
    )

    result = person_got.compute_similarity(person_expected)

    # got.address is BaseModel, expected.address is string -> similarity = 0.0
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_similarity_nested_one_missing_with_internal_fields() -> None:
    """Test similarity when nested model one missing and has internal fields."""

    class Address(BaseModel):
        street: str
        _internal: str = None  # Internal field

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person(name="John")  # address missing
    person_expected = Person.from_dict({"name": "Jane", "address": {"street": "X"}})

    result = person_got.compute_similarity(person_expected)

    # Got missing, expected present -> similarity = 0.0 for nested fields
    # Internal fields (_internal) should be skipped
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0


def test_similarity_nested_both_present_expected_not_basemodel() -> None:
    """Test similarity when nested Field is present but expected is not BaseModel instance."""

    class Address(BaseModel):
        street: str
        city: str
        _internal: str = None

    class Person(BaseModel):
        name: str
        address: Address

    # Create person_got with address as a Field containing a BaseModel
    person_got = Person(name="John")
    from cobjectric.field import Field
    from cobjectric.field_spec import FieldSpec

    address_got = Address(street="123 Main", city="NYC")
    person_got._fields["address"] = Field(
        name="address",
        type=Address,
        value=address_got,
        spec=FieldSpec(),
    )

    # Create person_expected with address as a Field containing a non-BaseModel
    person_expected = Person(name="Jane")
    person_expected._fields["address"] = Field(
        name="address",
        type=Address,
        value="string_not_basemodel",
        spec=FieldSpec(),
    )

    result = person_got.compute_similarity(person_expected)

    # got.address is Field with BaseModel value (not MissingValue)
    # expected.address is Field with string value (not BaseModel)
    # -> field is Field, not BaseModel, so doesn't enter first if
    # -> is_nested_model_type is True
    # -> field.value is not MissingValue and expected_value is not MissingValue
    # -> enters else branch
    # -> expected_nested is not BaseModel -> enters else branch
    # -> similarity = 0.0 for all nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_similarity_nested_field_both_present_both_basemodel() -> None:
    """Test similarity when nested Field is present and both are BaseModel instances."""

    class Address(BaseModel):
        street: str
        city: str
        _internal: str = None

    class Person(BaseModel):
        name: str
        address: Address

    # Create person_got with address as a Field containing a BaseModel
    person_got = Person(name="John")
    from cobjectric.field import Field
    from cobjectric.field_spec import FieldSpec

    address_got = Address(street="123 Main", city="NYC")
    person_got._fields["address"] = Field(
        name="address",
        type=Address,
        value=address_got,
        spec=FieldSpec(),
    )

    # Create person_expected with address as a Field containing a BaseModel
    person_expected = Person(name="Jane")
    address_expected = Address(street="456 Oak", city="Somewhere")
    person_expected._fields["address"] = Field(
        name="address",
        type=Address,
        value=address_expected,
        spec=FieldSpec(),
    )

    result = person_got.compute_similarity(person_expected)

    # got.address is Field with BaseModel value (not MissingValue)
    # expected.address is Field with BaseModel value (not MissingValue)
    # -> field is Field, not BaseModel, so doesn't enter first if
    # -> is_nested_model_type is True
    # -> field.value is not MissingValue and expected_value is not MissingValue
    # -> expected_nested is BaseModel
    # -> recursively compute similarity
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0  # Different values
    assert result.fields.address.fields.city.value == 0.0  # Different values
