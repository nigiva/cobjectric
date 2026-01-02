import pytest

from cobjectric import BaseModel, IncompatibleModelResultError, ModelResultCollection
from cobjectric.results import (
    AggregatedFieldResult,
    AggregatedFieldResultCollection,
    AggregatedModelResult,
    FieldResult,
    FieldResultCollection,
    ListResult,
    ModelResult,
    NestedListAggregatedResult,
)


def test_model_result_collection_add_incompatible_model_result() -> None:
    """Test ModelResultCollection.__add__ raises IncompatibleModelResultError."""

    class Person(BaseModel):
        name: str
        age: int

    class Product(BaseModel):
        title: str
        price: float

    person = Person(name="John", age=30)
    product = Product(title="Book", price=10.0)

    result1 = person.compute_fill_rate()
    result2 = product.compute_fill_rate()

    collection = ModelResultCollection([result1])

    with pytest.raises(IncompatibleModelResultError):
        _ = collection + result2


def test_model_result_collection_add_incompatible_collection() -> None:
    """Test ModelResultCollection.__add__ raises IncompatibleModelResultError."""

    class Person(BaseModel):
        name: str
        age: int

    class Product(BaseModel):
        title: str
        price: float

    person = Person(name="John", age=30)
    product = Product(title="Book", price=10.0)

    result1 = person.compute_fill_rate()
    result2 = product.compute_fill_rate()

    collection1 = ModelResultCollection([result1])
    collection2 = ModelResultCollection([result2])

    with pytest.raises(IncompatibleModelResultError):
        _ = collection1 + collection2


def test_model_result_collection_add_invalid_type() -> None:
    """Test ModelResultCollection.__add__ raises TypeError for invalid type."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()
    collection = ModelResultCollection([result])

    with pytest.raises(TypeError, match="Cannot add"):
        _ = collection + "invalid"  # type: ignore[operator]


def test_model_result_collection_mean_empty() -> None:
    """Test ModelResultCollection.mean() with empty collection."""
    collection = ModelResultCollection([])
    means = collection.mean()

    assert means == {}


def test_model_result_collection_std_empty() -> None:
    """Test ModelResultCollection.std() with empty collection."""
    collection = ModelResultCollection([])
    stds = collection.std()

    assert stds == {}


def test_model_result_collection_var_empty() -> None:
    """Test ModelResultCollection.var() with empty collection."""
    collection = ModelResultCollection([])
    vars_dict = collection.var()

    assert vars_dict == {}


def test_model_result_collection_min_empty() -> None:
    """Test ModelResultCollection.min() with empty collection."""
    collection = ModelResultCollection([])
    mins = collection.min()

    assert mins == {}


def test_model_result_collection_max_empty() -> None:
    """Test ModelResultCollection.max() with empty collection."""
    collection = ModelResultCollection([])
    maxs = collection.max()

    assert maxs == {}


def test_model_result_collection_quantile_invalid_q() -> None:
    """Test ModelResultCollection.quantile() raises ValueError for invalid q."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()
    collection = ModelResultCollection([result])

    with pytest.raises(ValueError, match="Quantile q must be between"):
        _ = collection.quantile(-0.1)

    with pytest.raises(ValueError, match="Quantile q must be between"):
        _ = collection.quantile(1.1)


def test_model_result_collection_quantile_empty() -> None:
    """Test ModelResultCollection.quantile() with empty collection."""
    collection = ModelResultCollection([])
    quantiles = collection.quantile(0.5)

    assert quantiles == {}


def test_model_result_collection_quantile_q_zero() -> None:
    """Test ModelResultCollection.quantile() with q=0.0."""

    class Person(BaseModel):
        score: float

    person1 = Person(score=0.1)
    person2 = Person(score=0.5)
    person3 = Person(score=0.9)

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()
    result3 = person3.compute_fill_rate()

    collection = ModelResultCollection([result1, result2, result3])
    quantiles = collection.quantile(0.0)

    assert "score" in quantiles
    assert quantiles["score"] == 1.0  # All have score=1.0 (filled)


def test_model_result_collection_quantile_q_one() -> None:
    """Test ModelResultCollection.quantile() with q=1.0."""

    class Person(BaseModel):
        score: float

    person1 = Person(score=0.1)
    person2 = Person(score=0.5)
    person3 = Person(score=0.9)

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()
    result3 = person3.compute_fill_rate()

    collection = ModelResultCollection([result1, result2, result3])
    quantiles = collection.quantile(1.0)

    assert "score" in quantiles
    assert quantiles["score"] == 1.0  # All have score=1.0 (filled)


def test_model_result_collection_quantile_no_values() -> None:
    """Test ModelResultCollection.quantile() with path having no values."""

    class Person(BaseModel):
        name: str
        items: list[str]

    # First person has items, second has empty list
    # Empty list doesn't produce any fields in flattened dict
    person1 = Person(name="John", items=["item1", "item2"])
    person2 = Person(name="Jane", items=[])

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()

    collection = ModelResultCollection([result1, result2])

    # Mock _flatten_to_dict to simulate a case where a field exists in first
    # but not in second (e.g., nested field from list that's empty in second)
    call_tracker = {"first": True}

    original_flatten = None
    try:
        from cobjectric import pandas_export

        original_flatten = pandas_export._flatten_to_dict

        def mock_flatten(result: ModelResult, prefix: str = "") -> dict[str, float]:
            if call_tracker["first"]:
                call_tracker["first"] = False
                # First call - return dict with extra field
                flattened = original_flatten(result, prefix)
                flattened["items.extra"] = 0.5
                return flattened
            else:
                # Subsequent calls - don't include extra field
                return original_flatten(result, prefix)

        pandas_export._flatten_to_dict = mock_flatten
        quantiles = collection.quantile(0.5)
    finally:
        if original_flatten:
            pandas_export._flatten_to_dict = original_flatten

    # name should be in quantiles
    assert "name" in quantiles
    # items.extra should not be in quantiles because it's only in first result
    assert "items.extra" not in quantiles


def test_field_result_repr() -> None:
    """Test FieldResult.__repr__."""
    field_result = FieldResult(value=0.5, weight=1.0)
    repr_str = repr(field_result)

    assert "FieldResult" in repr_str
    assert "0.5" in repr_str
    assert "1.0" in repr_str


def test_field_result_collection_repr() -> None:
    """Test FieldResultCollection.__repr__."""
    fields = {
        "name": FieldResult(value=1.0),
        "age": FieldResult(value=0.0),
    }
    collection = FieldResultCollection(fields)
    repr_str = repr(collection)

    assert "FieldResultCollection" in repr_str
    assert "name" in repr_str
    assert "age" in repr_str


def test_model_result_repr() -> None:
    """Test ModelResult.__repr__."""
    fields = {
        "name": FieldResult(value=1.0),
    }
    result = ModelResult(_fields=fields)
    repr_str = repr(result)

    assert "ModelResult" in repr_str
    assert "name" in repr_str


def test_aggregated_field_result_repr() -> None:
    """Test AggregatedFieldResult.__repr__."""
    agg_result = AggregatedFieldResult(_values=[1.0, 0.5], _weights=[1.0, 1.0])
    repr_str = repr(agg_result)

    assert "AggregatedFieldResult" in repr_str
    assert "[1.0, 0.5]" in repr_str


def test_aggregated_model_result_repr() -> None:
    """Test AggregatedModelResult.__repr__."""
    fields = {"name": FieldResult(value=1.0)}
    items = [ModelResult(_fields=fields), ModelResult(_fields=fields)]
    agg_result = AggregatedModelResult(_items=items)
    repr_str = repr(agg_result)

    assert "AggregatedModelResult" in repr_str
    assert "2" in repr_str


def test_nested_list_aggregated_result_repr() -> None:
    """Test NestedListAggregatedResult.__repr__."""
    fields = {"name": FieldResult(value=1.0)}
    items = [ModelResult(_fields=fields)]
    list_result = ListResult(_items=items)
    nested_result = NestedListAggregatedResult(lists=[list_result])
    repr_str = repr(nested_result)

    assert "NestedListAggregatedResult" in repr_str
    assert "1" in repr_str


def test_aggregated_field_result_collection_repr_empty() -> None:
    """Test AggregatedFieldResultCollection.__repr__ with empty items."""
    collection = AggregatedFieldResultCollection([])
    repr_str = repr(collection)

    assert "AggregatedFieldResultCollection()" == repr_str


def test_aggregated_field_result_collection_repr() -> None:
    """Test AggregatedFieldResultCollection.__repr__ with items."""
    fields = {"name": FieldResult(value=1.0)}
    items = [ModelResult(_fields=fields)]
    collection = AggregatedFieldResultCollection(items)
    repr_str = repr(collection)

    assert "AggregatedFieldResultCollection" in repr_str
    assert "name" in repr_str


def test_list_result_repr() -> None:
    """Test ListResult.__repr__."""
    fields = {"name": FieldResult(value=1.0)}
    items = [ModelResult(_fields=fields)]
    list_result = ListResult(_items=items, weight=2.0)
    repr_str = repr(list_result)

    assert "ListResult" in repr_str
    assert "1" in repr_str
    assert "2.0" in repr_str


def test_model_result_collection_repr() -> None:
    """Test ModelResultCollection.__repr__."""
    fields = {"name": FieldResult(value=1.0)}
    items = [ModelResult(_fields=fields), ModelResult(_fields=fields)]
    collection = ModelResultCollection(items)
    repr_str = repr(collection)

    assert "ModelResultCollection" in repr_str
    assert "2" in repr_str


def test_model_result_collection_len() -> None:
    """Test ModelResultCollection.__len__."""
    fields = {"name": FieldResult(value=1.0)}
    items = [ModelResult(_fields=fields), ModelResult(_fields=fields)]
    collection = ModelResultCollection(items)

    assert len(collection) == 2


def test_model_result_collection_std_single_value() -> None:
    """Test ModelResultCollection.std() with single value."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()
    collection = ModelResultCollection([result])
    stds = collection.std()

    assert "name" in stds
    assert stds["name"] == 0.0


def test_model_result_collection_var() -> None:
    """Test ModelResultCollection.var() method."""

    class Person(BaseModel):
        score: float

    person1 = Person(score=0.1)
    person2 = Person(score=0.5)
    person3 = Person(score=0.9)

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()
    result3 = person3.compute_fill_rate()

    collection = ModelResultCollection([result1, result2, result3])
    vars_dict = collection.var()

    assert "score" in vars_dict
    assert vars_dict["score"] == 0.0  # All have score=1.0 (filled), so variance is 0


def test_model_result_collection_var_single_value() -> None:
    """Test ModelResultCollection.var() with single value."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()
    collection = ModelResultCollection([result])
    vars_dict = collection.var()

    assert "name" in vars_dict
    assert vars_dict["name"] == 0.0
