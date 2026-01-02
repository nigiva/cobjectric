import typing as t
from unittest.mock import patch

import pytest

from cobjectric import (
    BaseModel,
    IncompatibleModelResultError,
    ModelResultCollection,
)

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def test_to_series_simple_model() -> None:
    """Test to_series() on a simple model."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person = Person(name="John", age=30, email="john@example.com")
    result = person.compute_fill_rate()
    series = result.to_series()

    assert isinstance(series, pd.Series)
    assert series["name"] == 1.0
    assert series["age"] == 1.0
    assert series["email"] == 1.0


def test_to_series_with_missing_fields() -> None:
    """Test to_series() with missing fields."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()
    series = result.to_series()

    assert isinstance(series, pd.Series)
    assert series["name"] == 1.0
    assert series["age"] == 1.0
    assert series["email"] == 0.0


def test_to_series_nested_model() -> None:
    """Test to_series() with nested models."""

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

    assert isinstance(series, pd.Series)
    assert series["name"] == 1.0
    assert series["address.street"] == 1.0
    assert series["address.city"] == 1.0


def test_to_series_with_list() -> None:
    """Test to_series() with list fields using aggregated_fields."""

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

    assert isinstance(series, pd.Series)
    # List fields use aggregated_fields.mean()
    assert "items.name" in series.index
    assert "items.price" in series.index


def test_model_result_addition() -> None:
    """Test adding two ModelResults."""

    class Person(BaseModel):
        name: str
        age: int

    person1 = Person(name="John", age=30)
    person2 = Person(name="Jane", age=25)

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()

    collection = result1 + result2

    assert isinstance(collection, ModelResultCollection)
    assert len(collection) == 2


def test_model_result_collection_addition() -> None:
    """Test adding ModelResult to ModelResultCollection."""

    class Person(BaseModel):
        name: str
        age: int

    person1 = Person(name="John", age=30)
    person2 = Person(name="Jane", age=25)
    person3 = Person(name="Bob", age=40)

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()
    result3 = person3.compute_fill_rate()

    collection = result1 + result2
    collection = collection + result3

    assert isinstance(collection, ModelResultCollection)
    assert len(collection) == 3


def test_model_result_collection_merge() -> None:
    """Test merging two ModelResultCollections."""

    class Person(BaseModel):
        name: str
        age: int

    person1 = Person(name="John", age=30)
    person2 = Person(name="Jane", age=25)
    person3 = Person(name="Bob", age=40)

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()
    result3 = person3.compute_fill_rate()

    collection1 = result1 + result2
    collection2 = ModelResultCollection([result3])
    merged = collection1 + collection2

    assert isinstance(merged, ModelResultCollection)
    assert len(merged) == 3


def test_incompatible_model_result_error() -> None:
    """Test that IncompatibleModelResultError is raised for different models."""

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

    with pytest.raises(IncompatibleModelResultError):
        _ = result1 + result2


def test_to_dataframe() -> None:
    """Test to_dataframe() on ModelResultCollection."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person1 = Person(name="John", age=30, email="john@example.com")
    person2 = Person(name="Jane", age=25)
    person3 = Person(name="Bob", age=40, email="bob@example.com")

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()
    result3 = person3.compute_fill_rate()

    collection = result1 + result2 + result3
    df = collection.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "name" in df.columns
    assert "age" in df.columns
    assert "email" in df.columns
    assert df.loc[0, "name"] == 1.0
    assert df.loc[1, "email"] == 0.0


def test_to_dataframe_nested_models() -> None:
    """Test to_dataframe() with nested models."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person1 = Person(
        name="John",
        address=Address(street="123 Main St", city="Anytown"),
    )
    person2 = Person(name="Jane", address=Address(street="456 Oak Ave", city=""))

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()

    collection = result1 + result2
    df = collection.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "name" in df.columns
    assert "address.street" in df.columns
    assert "address.city" in df.columns


def test_collection_mean() -> None:
    """Test mean() on ModelResultCollection."""

    class Person(BaseModel):
        name: str
        age: int

    person1 = Person(name="John", age=30)
    person2 = Person(name="Jane", age=25)
    person3 = Person(name="Bob", age=40)

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()
    result3 = person3.compute_fill_rate()

    collection = result1 + result2 + result3
    means = collection.mean()

    assert isinstance(means, dict)
    assert "name" in means
    assert "age" in means
    assert means["name"] == 1.0  # All have name
    assert means["age"] == 1.0  # All have age


def test_collection_std() -> None:
    """Test std() on ModelResultCollection."""

    class Person(BaseModel):
        score: float

    person1 = Person(score=10.0)
    person2 = Person(score=20.0)
    person3 = Person(score=30.0)

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()
    result3 = person3.compute_fill_rate()

    collection = result1 + result2 + result3
    stds = collection.std()

    assert isinstance(stds, dict)
    assert "score" in stds
    # All have score, so std should be 0.0
    assert stds["score"] == 0.0


def test_collection_min_max() -> None:
    """Test min() and max() on ModelResultCollection."""

    class Person(BaseModel):
        name: str
        age: int

    person1 = Person(name="John", age=30)
    person2 = Person(name="Jane", age=25)
    person3 = Person(name="Bob", age=40)

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()
    result3 = person3.compute_fill_rate()

    collection = result1 + result2 + result3
    mins = collection.min()
    maxs = collection.max()

    assert isinstance(mins, dict)
    assert isinstance(maxs, dict)
    assert mins["name"] == 1.0
    assert maxs["name"] == 1.0


def test_collection_quantile() -> None:
    """Test quantile() on ModelResultCollection."""

    class Person(BaseModel):
        score: float

    person1 = Person(score=10.0)
    person2 = Person(score=20.0)
    person3 = Person(score=30.0)

    result1 = person1.compute_fill_rate()
    result2 = person2.compute_fill_rate()
    result3 = person3.compute_fill_rate()

    collection = result1 + result2 + result3
    median = collection.quantile(0.5)

    assert isinstance(median, dict)
    assert "score" in median
    assert median["score"] == 1.0  # All have score


def test_to_series_without_pandas() -> None:
    """Test that to_series() raises ImportError when pandas is not installed."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()

    def mock_get_pandas() -> None:
        raise ImportError(
            "pandas is required for to_series() and to_dataframe(). "
            "Install it with: pip install cobjectric[pandas]"
        )

    with patch("cobjectric.pandas_export._get_pandas", side_effect=mock_get_pandas):
        with pytest.raises(ImportError, match="pandas is required"):
            _ = result.to_series()


def test_to_dataframe_without_pandas() -> None:
    """Test that to_dataframe() raises ImportError when pandas is not installed."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()
    collection = result + result

    def mock_get_pandas() -> None:
        raise ImportError(
            "pandas is required for to_series() and to_dataframe(). "
            "Install it with: pip install cobjectric[pandas]"
        )

    with patch("cobjectric.pandas_export._get_pandas", side_effect=mock_get_pandas):
        with pytest.raises(ImportError, match="pandas is required"):
            _ = collection.to_dataframe()


def test_check_pandas_available() -> None:
    """Test _check_pandas_available() function."""
    from cobjectric.pandas_export import _check_pandas_available

    # Should return True when pandas is available
    assert _check_pandas_available() is True

    # Test when pandas is not available
    original_import = __import__

    def mock_import(name: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        import importlib

        import cobjectric.pandas_export

        importlib.reload(cobjectric.pandas_export)
        result = cobjectric.pandas_export._check_pandas_available()
        assert result is False
        # Reload again to restore normal behavior
        importlib.reload(cobjectric.pandas_export)


def test_get_pandas_raises_import_error() -> None:
    """Test that _get_pandas() raises ImportError with proper message."""
    original_import = __import__

    def mock_import(name: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        import importlib

        import cobjectric.pandas_export

        importlib.reload(cobjectric.pandas_export)
        with pytest.raises(ImportError, match="pandas is required"):
            cobjectric.pandas_export._get_pandas()
        # Reload again to restore normal behavior
        importlib.reload(cobjectric.pandas_export)


def test_flatten_field_result_without_prefix() -> None:
    """Test _flatten_to_dict with FieldResult without prefix (edge case)."""
    from cobjectric.pandas_export import _flatten_to_dict
    from cobjectric.results import FieldResult

    field_result = FieldResult(value=0.5, weight=1.0)
    flattened = _flatten_to_dict(field_result, prefix="")

    assert flattened == {"value": 0.5}


def test_flatten_empty_list_result() -> None:
    """Test _flatten_to_dict with empty ListResult."""
    from cobjectric.pandas_export import _flatten_to_dict
    from cobjectric.results import ListResult

    list_result = ListResult(_items=[], weight=1.0)
    flattened = _flatten_to_dict(list_result, prefix="items")

    assert flattened == {}


def test_flatten_list_with_nested_model() -> None:
    """Test _flatten_to_dict with list containing nested models."""
    from cobjectric import BaseModel
    from cobjectric.pandas_export import _flatten_to_dict

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        addresses: list[Address]

    person = Person(
        name="John",
        addresses=[
            Address(street="123 Main St", city="Anytown"),
            Address(street="456 Oak Ave", city="Othertown"),
        ],
    )
    result = person.compute_fill_rate()

    # Test direct flattening of ListResult with nested models
    list_result = result.fields.addresses
    flattened = _flatten_to_dict(list_result, prefix="addresses")

    # Should have flattened nested model fields
    assert "addresses.street" in flattened
    assert "addresses.city" in flattened
    assert flattened["addresses.street"] == 1.0
    assert flattened["addresses.city"] == 1.0

    # Test via to_series as well
    series = result.to_series()
    assert "name" in series.index
    assert "addresses.street" in series.index
    assert "addresses.city" in series.index


def test_flatten_list_with_nested_model_in_list() -> None:
    """Test _flatten_to_dict with list of models containing nested models."""
    from cobjectric import BaseModel
    from cobjectric.pandas_export import _flatten_to_dict

    class Address(BaseModel):
        street: str
        city: str

    class Item(BaseModel):
        name: str
        address: Address

    class Order(BaseModel):
        items: list[Item]

    order = Order(
        items=[
            Item(name="Apple", address=Address(street="123 Main St", city="NYC")),
            Item(name="Banana", address=Address(street="456 Oak Ave", city="LA")),
        ],
    )
    result = order.compute_fill_rate()

    # Test direct flattening of ListResult where items have nested models
    list_result = result.fields.items
    flattened = _flatten_to_dict(list_result, prefix="items")

    # Should have flattened both simple fields and nested model fields
    assert "items.name" in flattened
    assert "items.address.street" in flattened
    assert "items.address.city" in flattened
    assert flattened["items.name"] == 1.0
    assert flattened["items.address.street"] == 1.0
    assert flattened["items.address.city"] == 1.0

    # Test via to_series as well
    series = result.to_series()
    assert "items.name" in series.index
    assert "items.address.street" in series.index
    assert "items.address.city" in series.index
