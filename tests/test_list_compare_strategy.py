import pytest

from cobjectric import (
    BaseModel,
    InvalidListCompareStrategyError,
    ListCompareStrategy,
    ListResult,
    Spec,
)


def test_similarity_pairwise_default() -> None:
    """Test that pairwise is the default strategy for similarity."""

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

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0
    assert result.fields.items[1].fields.price.value == 1.0


def test_similarity_pairwise_explicit() -> None:
    """Test pairwise strategy explicitly for similarity."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.PAIRWISE)

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

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0
    assert result.fields.items[1].fields.name.value == 0.0
    assert result.fields.items[1].fields.price.value == 0.0


def test_similarity_pairwise_different_lengths() -> None:
    """Test pairwise strategy with different list lengths."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy="pairwise")

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
            ],
        }
    )
    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 1
    assert result.fields.items[0].fields.name.value == 1.0


def test_similarity_levenshtein_same_order() -> None:
    """Test levenshtein strategy when items are in the same order."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

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

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0
    assert result.fields.items[1].fields.price.value == 1.0


def test_similarity_levenshtein_with_insertion() -> None:
    """Test levenshtein strategy with an insertion in the middle.

    Levenshtein preserves relative order, so it should align items correctly
    even when there's an extra item in between.

    got:      [Apple, Cherry, Banana]
    expected: [Apple, Banana]

    Best alignment: (0,0) Apple-Apple, (2,1) Banana-Banana (skip Cherry)
    """

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Cherry"},
                {"name": "Banana"},
            ],
        }
    )
    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0


def test_similarity_levenshtein_reversed_lists() -> None:
    """Test levenshtein strategy with completely reversed lists.

    Levenshtein PRESERVES relative order, so it cannot align both items
    when lists are completely reversed.

    got:      [Apple, Banana]
    expected: [Banana, Apple]

    Levenshtein can only align ONE item (either Apple-Apple OR Banana-Banana)
    because aligning both would violate the order constraint.
    """

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

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
                {"name": "Banana", "price": 0.5},
                {"name": "Apple", "price": 1.0},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    # Levenshtein can only align one item for reversed lists
    assert len(result.fields.items) == 1
    # The aligned item should have perfect similarity
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0


def test_similarity_levenshtein_different_lengths() -> None:
    """Test levenshtein strategy with different list lengths."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy="levenshtein")

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
                {"name": "Cherry"},
            ],
        }
    )
    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "Banana"},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 1
    assert result.fields.items[0].fields.name.value == 1.0


def test_similarity_optimal_assignment_basic() -> None:
    """Test optimal assignment strategy for similarity.

    Optimal assignment (Hungarian algorithm) finds the best one-to-one mapping
    WITHOUT order constraints. It CAN align items even when lists are reversed.

    got:      [Apple, Banana]
    expected: [Banana, Apple]

    Optimal alignment: (0,1) Apple-Apple, (1,0) Banana-Banana
    """

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item] = Spec(
            list_compare_strategy=ListCompareStrategy.OPTIMAL_ASSIGNMENT
        )

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
                {"name": "Banana", "price": 0.5},
                {"name": "Apple", "price": 1.0},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0
    assert result.fields.items[1].fields.price.value == 1.0


def test_similarity_optimal_assignment_different_lengths() -> None:
    """Test optimal assignment strategy with different list lengths."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy="optimal_assignment")

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
            ],
        }
    )
    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "Banana"},
                {"name": "Cherry"},
                {"name": "Date"},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2


def test_accuracy_pairwise_default() -> None:
    """Test that pairwise is the default strategy for fill rate accuracy."""

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

    result = order_got.compute_fill_rate_accuracy(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0
    assert result.fields.items[1].fields.price.value == 1.0


def test_accuracy_pairwise_explicit() -> None:
    """Test pairwise strategy explicitly for fill rate accuracy."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.PAIRWISE)

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana"},
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

    result = order_got.compute_fill_rate_accuracy(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0
    assert result.fields.items[1].fields.price.value == 0.0


def test_accuracy_levenshtein_same_order() -> None:
    """Test levenshtein strategy for fill rate accuracy with same order."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

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

    result = order_got.compute_fill_rate_accuracy(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0
    assert result.fields.items[1].fields.price.value == 1.0


def test_accuracy_levenshtein_reversed() -> None:
    """Test levenshtein strategy for fill rate accuracy with reversed lists.

    For fill_rate_accuracy, Levenshtein uses `same_state_fill_rate_accuracy`
    which compares if fields are filled (not values). Since all items have
    the same filled state (all fields present), ALL matches have score 1.0.

    This means the algorithm treats (0,0) and (0,1) as equally good matches,
    so it can align both items by matching (0,0) and (1,1).

    This is different from similarity which compares actual VALUES.
    """

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

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
                {"name": "Banana", "price": 0.5},
                {"name": "Apple", "price": 1.0},
            ],
        }
    )

    result = order_got.compute_fill_rate_accuracy(order_expected)

    assert isinstance(result.fields.items, ListResult)
    # For accuracy, all matches are equal (score 1.0) so it aligns both items
    assert len(result.fields.items) == 2
    # All accuracy values should be 1.0 since all fields are filled in both items
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0
    assert result.fields.items[1].fields.price.value == 1.0


def test_accuracy_optimal_assignment_basic() -> None:
    """Test optimal assignment strategy for fill rate accuracy."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item] = Spec(
            list_compare_strategy=ListCompareStrategy.OPTIMAL_ASSIGNMENT
        )

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
                {"name": "Banana", "price": 0.5},
                {"name": "Apple", "price": 1.0},
            ],
        }
    )

    result = order_got.compute_fill_rate_accuracy(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0
    assert result.fields.items[1].fields.price.value == 1.0


def test_list_compare_strategy_on_non_list_raises() -> None:
    """Test that using list_compare_strategy on non-list field raises error."""

    class Person(BaseModel):
        name: str = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    with pytest.raises(InvalidListCompareStrategyError):
        person_got.compute_similarity(person_expected)


def test_list_compare_strategy_on_primitive_list_raises() -> None:
    """Test that using list_compare_strategy on list[Primitive] raises error."""

    class Person(BaseModel):
        tags: list[str] = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

    person_got = Person(tags=["python"])
    person_expected = Person(tags=["rust"])

    with pytest.raises(InvalidListCompareStrategyError):
        person_got.compute_similarity(person_expected)


def test_list_compare_strategy_empty_lists() -> None:
    """Test list compare strategy with empty lists."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

    order_got = Order.from_dict({"items": []})
    order_expected = Order.from_dict({"items": []})

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 0


def test_list_compare_strategy_one_empty_list() -> None:
    """Test list compare strategy when one list is empty."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item] = Spec(
            list_compare_strategy=ListCompareStrategy.OPTIMAL_ASSIGNMENT
        )

    order_got = Order.from_dict({"items": [{"name": "Apple"}]})
    order_expected = Order.from_dict({"items": []})

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 0


def test_list_compare_strategy_missing_list() -> None:
    """Test list compare strategy when list is MissingValue."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.PAIRWISE)

    order_got = Order()
    order_expected = Order.from_dict({"items": [{"name": "Apple"}]})

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 0


def test_list_compare_strategy_string_value() -> None:
    """Test that list_compare_strategy accepts string values."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy="pairwise")

    order_got = Order.from_dict({"items": [{"name": "Apple"}]})
    order_expected = Order.from_dict({"items": [{"name": "Apple"}]})

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 1
    assert result.fields.items[0].fields.name.value == 1.0


def test_list_compare_strategy_enum_value() -> None:
    """Test that list_compare_strategy accepts ListCompareStrategy enum values."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.PAIRWISE)

    order_got = Order.from_dict({"items": [{"name": "Apple"}]})
    order_expected = Order.from_dict({"items": [{"name": "Apple"}]})

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 1
    assert result.fields.items[0].fields.name.value == 1.0


def test_list_compare_strategy_invalid_string_raises() -> None:
    """Test that invalid string value for list_compare_strategy raises error."""

    with pytest.raises(ValueError):
        Spec(list_compare_strategy="invalid_strategy")


def test_list_compare_strategy_invalid_type_raises() -> None:
    """Test that invalid type for list_compare_strategy raises error."""

    with pytest.raises(
        ValueError, match="list_compare_strategy must be str or ListCompareStrategy"
    ):
        Spec(list_compare_strategy=42)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError, match="list_compare_strategy must be str or ListCompareStrategy"
    ):
        Spec(list_compare_strategy=[])  # type: ignore[arg-type]

    with pytest.raises(
        ValueError, match="list_compare_strategy must be str or ListCompareStrategy"
    ):
        Spec(list_compare_strategy={})  # type: ignore[arg-type]


def test_list_compare_strategy_nested_models() -> None:
    """Test list compare strategy with nested models in list items.

    Using optimal_assignment since it can align items regardless of order.
    """

    class Address(BaseModel):
        city: str

    class Item(BaseModel):
        name: str
        address: Address

    class Order(BaseModel):
        items: list[Item] = Spec(
            list_compare_strategy=ListCompareStrategy.OPTIMAL_ASSIGNMENT
        )

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "address": {"city": "NYC"}},
                {"name": "Banana", "address": {"city": "LA"}},
            ],
        }
    )
    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "Banana", "address": {"city": "LA"}},
                {"name": "Apple", "address": {"city": "NYC"}},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.address.fields.city.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0
    assert result.fields.items[1].fields.address.fields.city.value == 1.0


def test_list_compare_strategy_nested_models_levenshtein_same_order() -> None:
    """Test levenshtein strategy with nested models when order is preserved."""

    class Address(BaseModel):
        city: str

    class Item(BaseModel):
        name: str
        address: Address

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "address": {"city": "NYC"}},
                {"name": "Banana", "address": {"city": "LA"}},
            ],
        }
    )
    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "address": {"city": "NYC"}},
                {"name": "Banana", "address": {"city": "LA"}},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.address.fields.city.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0
    assert result.fields.items[1].fields.address.fields.city.value == 1.0


def test_list_compare_strategy_aggregated_access() -> None:
    """Test that aggregated access works with list compare strategies."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item] = Spec(
            list_compare_strategy=ListCompareStrategy.OPTIMAL_ASSIGNMENT
        )

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
                {"name": "Banana", "price": 0.5},
                {"name": "Apple", "price": 1.0},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items.aggregated_fields.name.values == [1.0, 1.0]
    assert result.fields.items.aggregated_fields.price.values == [1.0, 1.0]


def test_list_compare_strategy_levenshtein_partial_match() -> None:
    """Test levenshtein strategy with partial matches.

    got:      [A, B, C, D]
    expected: [A, X, C, Y]

    Best alignment preserving order: (0,0) A-A, (2,2) C-C
    Skips B, D from got and X, Y from expected.
    """

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item] = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
                {"name": "D"},
            ],
        }
    )
    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "A"},
                {"name": "X"},
                {"name": "C"},
                {"name": "Y"},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    # Should align A-A and C-C (preserving order, skipping non-matching items)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0


def test_list_compare_strategy_optimal_assignment_partial_match() -> None:
    """Test optimal assignment with partial matches and different lengths."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item] = Spec(
            list_compare_strategy=ListCompareStrategy.OPTIMAL_ASSIGNMENT
        )

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "A"},
                {"name": "B"},
            ],
        }
    )
    order_expected = Order.from_dict(
        {
            "items": [
                {"name": "X"},
                {"name": "B"},
                {"name": "A"},
            ],
        }
    )

    result = order_got.compute_similarity(order_expected)

    assert isinstance(result.fields.items, ListResult)
    # Should optimally match: A-A and B-B (both with similarity 1.0)
    assert len(result.fields.items) == 2
    # Both should have perfect similarity due to optimal matching
    similarities = [result.fields.items[i].fields.name.value for i in range(2)]
    assert sorted(similarities) == [1.0, 1.0]


def test_list_compare_strategy_on_non_list_accuracy_raises() -> None:
    """Test that using list_compare_strategy on non-list field raises error for accuracy."""

    class Person(BaseModel):
        name: str = Spec(list_compare_strategy=ListCompareStrategy.LEVENSHTEIN)

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    with pytest.raises(InvalidListCompareStrategyError):
        person_got.compute_fill_rate_accuracy(person_expected)


def test_list_compare_strategy_on_primitive_list_accuracy_raises() -> None:
    """Test that using list_compare_strategy on list[Primitive] raises error for accuracy."""

    class Person(BaseModel):
        tags: list[str] = Spec(
            list_compare_strategy=ListCompareStrategy.OPTIMAL_ASSIGNMENT
        )

    person_got = Person(tags=["python"])
    person_expected = Person(tags=["rust"])

    with pytest.raises(InvalidListCompareStrategyError):
        person_got.compute_fill_rate_accuracy(person_expected)


def test_optimal_assignment_scipy_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ImportError is raised when scipy is not available."""
    import builtins
    from unittest.mock import MagicMock

    from cobjectric.list_compare import align_optimal_assignment
    from cobjectric.results import ModelResult

    original_import = builtins.__import__

    def mock_import(
        name: str,
        globals_dict: dict | None = None,
        locals_dict: dict | None = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> object:
        if name == "scipy.optimize":
            raise ImportError("No module named 'scipy'")
        return original_import(name, globals_dict, locals_dict, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    got_list = [MagicMock()]
    expected_list = [MagicMock()]

    def compute_sim(got: object, exp: object) -> ModelResult:
        return MagicMock(mean=lambda: 1.0)

    with pytest.raises(ImportError, match="scipy is required"):
        align_optimal_assignment(got_list, expected_list, compute_sim)
