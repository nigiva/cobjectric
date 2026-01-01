import typing as t

import pytest

from cobjectric import (
    BaseModel,
    DuplicateFillRateFuncError,
    FillRateAggregatedFieldCollection,
    FillRateAggregatedFieldResult,
    FillRateAggregatedModelResult,
    FillRateFieldResult,
    FillRateListResult,
    FillRateModelResult,
    FillRateNestedListAggregatedResult,
    InvalidAggregatedFieldError,
    InvalidFillRateValueError,
    InvalidWeightError,
    MissingValue,
    Spec,
    fill_rate_func,
)
from cobjectric.field import Field


def test_default_fill_rate_func_with_value() -> None:
    """Test that default fill_rate_func returns 1.0 when value is present."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John Doe", age=30)
    result = person.compute_fill_rate()

    assert result.fields.name.value == 1.0
    assert result.fields.age.value == 1.0


def test_default_fill_rate_func_with_missing_value() -> None:
    """Test that default fill_rate_func returns 0.0 when value is MissingValue."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John Doe")
    result = person.compute_fill_rate()

    assert result.fields.name.value == 1.0
    assert result.fields.age.value == 0.0


def test_spec_fill_rate_func_custom() -> None:
    """Test that Spec(fill_rate_func=...) works correctly."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: len(x) / 100)
        age: int

    person = Person(name="John Doe", age=30)
    result = person.compute_fill_rate()

    assert result.fields.name.value == 8.0 / 100  # len("John Doe") = 8
    assert result.fields.age.value == 1.0


def test_decorator_fill_rate_func_single_field() -> None:
    """Test that @fill_rate_func works with a single field."""

    class Person(BaseModel):
        name: str
        age: int

        @fill_rate_func("name")
        def fill_rate_name(x: t.Any) -> float:
            return len(x) / 100 if x is not MissingValue else 0.0

    person = Person(name="John Doe", age=30)
    result = person.compute_fill_rate()

    assert result.fields.name.value == 8.0 / 100
    assert result.fields.age.value == 1.0


def test_decorator_fill_rate_func_multiple_fields() -> None:
    """Test that @fill_rate_func works with multiple fields."""

    class Person(BaseModel):
        name: str
        email: str
        age: int

        @fill_rate_func("name", "email")
        def fill_rate_name_email(x: t.Any) -> float:
            return len(x) / 100 if x is not MissingValue else 0.0

    person = Person(name="John", email="john@example.com", age=30)
    result = person.compute_fill_rate()

    assert result.fields.name.value == 4.0 / 100
    assert result.fields.email.value == 16.0 / 100
    assert result.fields.age.value == 1.0


def test_decorator_fill_rate_func_pattern() -> None:
    """Test that @fill_rate_func supports glob patterns."""

    class Person(BaseModel):
        name_first: str
        name_last: str
        age: int

        @fill_rate_func("name_*")
        def fill_rate_name_fields(x: t.Any) -> float:
            return len(x) / 100 if x is not MissingValue else 0.0

    person = Person(name_first="John", name_last="Doe", age=30)
    result = person.compute_fill_rate()

    assert result.fields.name_first.value == 4.0 / 100
    assert result.fields.name_last.value == 3.0 / 100
    assert result.fields.age.value == 1.0


def test_duplicate_fill_rate_func_raises_error() -> None:
    """Test that multiple fill_rate_func for same field raises DuplicateFillRateFuncError."""

    class Person(BaseModel):
        name: str

        @fill_rate_func("name")
        def fill_rate_name1(x: t.Any) -> float:
            return 0.5

        @fill_rate_func("name")
        def fill_rate_name2(x: t.Any) -> float:
            return 0.6

    person = Person(name="John")
    with pytest.raises(DuplicateFillRateFuncError):
        person.compute_fill_rate()


def test_spec_and_decorator_same_field_raises_error() -> None:
    """Test that Spec + decorator on same field raises DuplicateFillRateFuncError."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.5)

        @fill_rate_func("name")
        def fill_rate_name(x: t.Any) -> float:
            return 0.6

    person = Person(name="John")
    with pytest.raises(DuplicateFillRateFuncError):
        person.compute_fill_rate()


def test_fill_rate_must_be_float() -> None:
    """Test that fill_rate_func returning non-float raises InvalidFillRateValueError."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: "not a float")

    person = Person(name="John")
    with pytest.raises(InvalidFillRateValueError):
        person.compute_fill_rate()


def test_fill_rate_must_be_between_0_and_1() -> None:
    """Test that fill_rate_func returning value outside [0,1] raises InvalidFillRateValueError."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 1.5)

    person = Person(name="John")
    with pytest.raises(InvalidFillRateValueError):
        person.compute_fill_rate()

    class Person2(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: -0.1)

    person2 = Person2(name="John")
    with pytest.raises(InvalidFillRateValueError):
        person2.compute_fill_rate()


def test_fill_rate_boundary_values() -> None:
    """Test that fill_rate_func accepts 0.0 and 1.0 as valid values."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.0)
        age: int = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    assert result.fields.name.value == 0.0
    assert result.fields.age.value == 1.0


def test_compute_fill_rate_returns_model_result() -> None:
    """Test that compute_fill_rate returns FillRateModelResult."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John Doe", age=30)
    result = person.compute_fill_rate()

    assert isinstance(result, FillRateModelResult)


def test_fill_rate_field_access() -> None:
    """Test that result.fields.name returns FillRateFieldResult."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John Doe", age=30)
    result = person.compute_fill_rate()

    assert isinstance(result.fields.name, FillRateFieldResult)
    assert result.fields.name.value == 1.0


def test_fill_rate_mean() -> None:
    """Test that result.mean() calculates the mean."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person = Person(name="John Doe", age=30)
    result = person.compute_fill_rate()

    # name=1.0, age=1.0, email=0.0 -> mean = 2/3
    assert result.mean() == pytest.approx(2.0 / 3.0)


def test_fill_rate_max() -> None:
    """Test that result.max() returns the maximum value."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.3)
        age: int = Spec(fill_rate_func=lambda x: 0.8)
        email: str = Spec(fill_rate_func=lambda x: 0.5)

    person = Person(name="John", age=30, email="john@example.com")
    result = person.compute_fill_rate()

    assert result.max() == 0.8


def test_fill_rate_min() -> None:
    """Test that result.min() returns the minimum value."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.3)
        age: int = Spec(fill_rate_func=lambda x: 0.8)
        email: str = Spec(fill_rate_func=lambda x: 0.5)

    person = Person(name="John", age=30, email="john@example.com")
    result = person.compute_fill_rate()

    assert result.min() == 0.3


def test_fill_rate_std() -> None:
    """Test that result.std() calculates the standard deviation."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.0)
        age: int = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    # mean = 0.5, std = sqrt(((0.0-0.5)^2 + (1.0-0.5)^2) / 2) = sqrt(0.25) = 0.5
    assert result.std() == pytest.approx(0.5)


def test_fill_rate_var() -> None:
    """Test that result.var() calculates the variance."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.0)
        age: int = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    # variance = 0.25
    assert result.var() == pytest.approx(0.25)


def test_fill_rate_quantile_25() -> None:
    """Test that result.quantile(0.25) works correctly."""

    class Person(BaseModel):
        a: str = Spec(fill_rate_func=lambda x: 0.0)
        b: str = Spec(fill_rate_func=lambda x: 0.25)
        c: str = Spec(fill_rate_func=lambda x: 0.5)
        d: str = Spec(fill_rate_func=lambda x: 0.75)
        e: str = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(a="a", b="b", c="c", d="d", e="e")
    result = person.compute_fill_rate()

    assert result.quantile(0.25) == pytest.approx(0.25)


def test_fill_rate_quantile_50() -> None:
    """Test that result.quantile(0.50) works correctly."""

    class Person(BaseModel):
        a: str = Spec(fill_rate_func=lambda x: 0.0)
        b: str = Spec(fill_rate_func=lambda x: 0.5)
        c: str = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(a="a", b="b", c="c")
    result = person.compute_fill_rate()

    assert result.quantile(0.50) == pytest.approx(0.5)


def test_fill_rate_quantile_75() -> None:
    """Test that result.quantile(0.75) works correctly."""

    class Person(BaseModel):
        a: str = Spec(fill_rate_func=lambda x: 0.0)
        b: str = Spec(fill_rate_func=lambda x: 0.25)
        c: str = Spec(fill_rate_func=lambda x: 0.5)
        d: str = Spec(fill_rate_func=lambda x: 0.75)
        e: str = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(a="a", b="b", c="c", d="d", e="e")
    result = person.compute_fill_rate()

    assert result.quantile(0.75) == pytest.approx(0.75)


def test_fill_rate_quantile_90() -> None:
    """Test that result.quantile(0.90) works correctly."""

    class Person(BaseModel):
        a: str = Spec(fill_rate_func=lambda x: 0.0)
        b: str = Spec(fill_rate_func=lambda x: 0.5)
        c: str = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(a="a", b="b", c="c")
    result = person.compute_fill_rate()

    # With values [0.0, 0.5, 1.0], quantile(0.90) interpolates between 0.5 and 1.0
    # index = 0.90 * 2 = 1.8, so between index 1 and 2
    # weight = 0.8, so 0.5 * 0.2 + 1.0 * 0.8 = 0.9
    assert result.quantile(0.90) == pytest.approx(0.9)


def test_fill_rate_quantile_95() -> None:
    """Test that result.quantile(0.95) works correctly."""

    class Person(BaseModel):
        a: str = Spec(fill_rate_func=lambda x: 0.0)
        b: str = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(a="a", b="b")
    result = person.compute_fill_rate()

    # With values [0.0, 1.0], quantile(0.95) interpolates
    # index = 0.95 * 1 = 0.95, so between index 0 and 1
    # weight = 0.95, so 0.0 * 0.05 + 1.0 * 0.95 = 0.95
    assert result.quantile(0.95) == pytest.approx(0.95)


def test_fill_rate_quantile_99() -> None:
    """Test that result.quantile(0.99) works correctly."""

    class Person(BaseModel):
        a: str = Spec(fill_rate_func=lambda x: 0.0)
        b: str = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(a="a", b="b")
    result = person.compute_fill_rate()

    # With values [0.0, 1.0], quantile(0.99) interpolates
    # index = 0.99 * 1 = 0.99, so between index 0 and 1
    # weight = 0.99, so 0.0 * 0.01 + 1.0 * 0.99 = 0.99
    assert result.quantile(0.99) == pytest.approx(0.99)


def test_fill_rate_quantile_100() -> None:
    """Test that result.quantile(1.0) works correctly."""

    class Person(BaseModel):
        a: str = Spec(fill_rate_func=lambda x: 0.0)
        b: str = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(a="a", b="b")
    result = person.compute_fill_rate()

    assert result.quantile(1.0) == pytest.approx(1.0)


def test_fill_rate_quantile_0() -> None:
    """Test that result.quantile(0.0) works correctly."""

    class Person(BaseModel):
        a: str = Spec(fill_rate_func=lambda x: 0.0)
        b: str = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(a="a", b="b")
    result = person.compute_fill_rate()

    assert result.quantile(0.0) == pytest.approx(0.0)


def test_fill_rate_nested_model() -> None:
    """Test that nested model returns FillRateModelResult in fields."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {
            "name": "John Doe",
            "address": {"street": "123 Main St", "city": "Anytown"},
        }
    )
    result = person.compute_fill_rate()

    assert isinstance(result.fields.name, FillRateFieldResult)
    assert isinstance(result.fields.address, FillRateModelResult)
    assert isinstance(result.fields.address.fields.street, FillRateFieldResult)
    assert isinstance(result.fields.address.fields.city, FillRateFieldResult)


def test_fill_rate_deeply_nested() -> None:
    """Test that deeply nested models work correctly."""

    class Country(BaseModel):
        name: str
        code: str

    class Address(BaseModel):
        street: str
        country: Country

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {
            "name": "John Doe",
            "address": {
                "street": "123 Main St",
                "country": {"name": "USA", "code": "US"},
            },
        }
    )
    result = person.compute_fill_rate()

    assert isinstance(result.fields.name, FillRateFieldResult)
    assert isinstance(result.fields.address, FillRateModelResult)
    assert isinstance(result.fields.address.fields.street, FillRateFieldResult)
    assert isinstance(result.fields.address.fields.country, FillRateModelResult)
    assert isinstance(
        result.fields.address.fields.country.fields.name, FillRateFieldResult
    )


def test_fill_rate_nested_missing_model() -> None:
    """Test that nested model MissingValue results in all fields at 0.0."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person(name="John Doe")
    result = person.compute_fill_rate()

    assert result.fields.name.value == 1.0
    assert isinstance(result.fields.address, FillRateModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_fill_rate_empty_model() -> None:
    """Test that empty model works correctly."""

    class EmptyModel(BaseModel):
        pass

    instance = EmptyModel()
    result = instance.compute_fill_rate()

    assert isinstance(result, FillRateModelResult)
    assert result.mean() == 0.0  # No fields, mean is 0
    assert result.max() == 0.0  # No fields, max is 0
    assert result.min() == 0.0  # No fields, min is 0
    assert result.std() == 0.0  # No fields, std is 0
    assert result.var() == 0.0  # No fields, var is 0
    assert result.quantile(0.5) == 0.0  # No fields, quantile is 0


def test_fill_rate_all_missing() -> None:
    """Test that all fields MissingValue results in mean = 0.0."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person = Person()
    result = person.compute_fill_rate()

    assert result.mean() == 0.0
    assert result.fields.name.value == 0.0
    assert result.fields.age.value == 0.0
    assert result.fields.email.value == 0.0


def test_fill_rate_all_present() -> None:
    """Test that all fields present results in mean = 1.0."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person = Person(name="John", age=30, email="john@example.com")
    result = person.compute_fill_rate()

    assert result.mean() == 1.0


def test_fill_rate_applies_to_normalized_value() -> None:
    """Test that fill_rate_func applies to normalized value."""

    class Person(BaseModel):
        name: str = Spec(
            normalizer=lambda x: x.lower(),
            fill_rate_func=lambda x: len(x) / 100,
        )

    person = Person(name="JOHN DOE")
    result = person.compute_fill_rate()

    # Normalized value is "john doe" (len=8), not "JOHN DOE" (len=8)
    assert result.fields.name.value == 8.0 / 100


def test_fill_rate_with_missing_value_in_func() -> None:
    """Test that fill_rate_func receives MissingValue when field is missing."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.0 if x is MissingValue else 1.0)

    person = Person()
    result = person.compute_fill_rate()

    assert result.fields.name.value == 0.0


def test_fill_rate_int_return_value() -> None:
    """Test that fill_rate_func returning int (0 or 1) is accepted."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 1 if x is not MissingValue else 0)

    person = Person(name="John")
    result = person.compute_fill_rate()

    assert result.fields.name.value == 1.0


def test_fill_rate_repr() -> None:
    """Test that FillRateFieldResult and FillRateModelResult have proper repr."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()

    assert "FillRateFieldResult" in repr(result.fields.name)
    assert "FillRateModelResult" in repr(result)


def test_fill_rate_field_access_error() -> None:
    """Test that accessing non-existent field raises AttributeError."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()

    with pytest.raises(AttributeError):
        _ = result.fields.non_existent


def test_fill_rate_statistics_with_single_value() -> None:
    """Test statistics with single field."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.5)

    person = Person(name="John")
    result = person.compute_fill_rate()

    assert result.mean() == 0.5
    assert result.max() == 0.5
    assert result.min() == 0.5
    assert result.std() == 0.0
    assert result.var() == 0.0
    assert result.quantile(0.5) == 0.5


def test_fill_rate_field_collection_iter() -> None:
    """Test that FillRateFieldCollection can be iterated."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    fields_list = list(result.fields)
    assert len(fields_list) == 2
    assert all(isinstance(f, FillRateFieldResult) for f in fields_list)


def test_fill_rate_collect_all_values_recursive() -> None:
    """Test that _collect_all_values works recursively for nested models."""

    class Address(BaseModel):
        street: str = Spec(fill_rate_func=lambda x: 0.5)
        city: str = Spec(fill_rate_func=lambda x: 0.7)

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.3)
        address: Address

    person = Person.from_dict(
        {
            "name": "John",
            "address": {"street": "123 Main St", "city": "Anytown"},
        }
    )
    result = person.compute_fill_rate()

    # Should collect all values: [0.3, 0.5, 0.7]
    values = result._collect_all_values()
    assert set(values) == {0.3, 0.5, 0.7}
    assert len(values) == 3


def test_fill_rate_quantile_invalid_value() -> None:
    """Test that quantile raises ValueError for invalid q."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()

    with pytest.raises(ValueError, match="Quantile q must be between 0.0 and 1.0"):
        result.quantile(1.5)

    with pytest.raises(ValueError, match="Quantile q must be between 0.0 and 1.0"):
        result.quantile(-0.1)


def test_fill_rate_std_with_two_values() -> None:
    """Test that std works correctly with two values."""

    class Person(BaseModel):
        a: str = Spec(fill_rate_func=lambda x: 0.0)
        b: str = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(a="a", b="b")
    result = person.compute_fill_rate()

    # mean = 0.5, std = sqrt(((0.0-0.5)^2 + (1.0-0.5)^2) / 2) = sqrt(0.25) = 0.5
    assert result.std() == pytest.approx(0.5)


def test_fill_rate_var_with_two_values() -> None:
    """Test that var works correctly with two values."""

    class Person(BaseModel):
        a: str = Spec(fill_rate_func=lambda x: 0.0)
        b: str = Spec(fill_rate_func=lambda x: 1.0)

    person = Person(a="a", b="b")
    result = person.compute_fill_rate()

    # variance = 0.25
    assert result.var() == pytest.approx(0.25)


def test_fill_rate_nested_missing_model_with_private_field() -> None:
    """Test that private fields in missing nested models are skipped."""

    class Address(BaseModel):
        street: str
        city: str
        _private_field: str  # Private field

    class Person(BaseModel):
        name: str
        address: Address

    person = Person(name="John Doe")  # address is missing
    result = person.compute_fill_rate()

    # Private field should be skipped
    assert "street" in result.fields.address._fields
    assert "city" in result.fields.address._fields
    assert "_private_field" not in result.fields.address._fields


def test_fill_rate_weight_default_1() -> None:
    """Test that weight defaults to 1.0."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    assert result.fields.name.weight == 1.0
    assert result.fields.age.weight == 1.0


def test_fill_rate_weight_in_spec() -> None:
    """Test that weight can be set in Spec()."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_weight=2.0)
        age: int = Spec(fill_rate_weight=0.5)

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    assert result.fields.name.weight == 2.0
    assert result.fields.age.weight == 0.5


def test_fill_rate_weight_in_decorator() -> None:
    """Test that weight can be set in @fill_rate_func decorator."""

    class Person(BaseModel):
        name: str
        age: int

        @fill_rate_func("name", weight=2.0)
        def fill_rate_name(x: t.Any) -> float:
            return 1.0 if x is not MissingValue else 0.0

        @fill_rate_func("age", weight=0.5)
        def fill_rate_age(x: t.Any) -> float:
            return 1.0 if x is not MissingValue else 0.0

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    assert result.fields.name.weight == 2.0
    assert result.fields.age.weight == 0.5


def test_fill_rate_weight_decorator_overrides_spec() -> None:
    """Test that decorator weight overrides Spec weight."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_weight=1.0)

        @fill_rate_func("name", weight=2.0)
        def fill_rate_name(x: t.Any) -> float:
            return 1.0 if x is not MissingValue else 0.0

    person = Person(name="John")
    result = person.compute_fill_rate()

    # Decorator weight (2.0) should override Spec weight (1.0)
    assert result.fields.name.weight == 2.0


def test_fill_rate_mean_weighted_simple() -> None:
    """Test weighted mean with different weights."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.5, fill_rate_weight=2.0)
        age: int = Spec(fill_rate_func=lambda x: 1.0, fill_rate_weight=1.0)

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    # Weighted mean: (0.5 * 2.0 + 1.0 * 1.0) / (2.0 + 1.0) = 2.0 / 3.0 = 0.666...
    assert result.mean() == pytest.approx(2.0 / 3.0)


def test_fill_rate_mean_weighted_zero_weight() -> None:
    """Test weighted mean with zero weight."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.5, fill_rate_weight=0.0)
        age: int = Spec(fill_rate_func=lambda x: 1.0, fill_rate_weight=1.0)

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    # Weighted mean: (0.5 * 0.0 + 1.0 * 1.0) / (0.0 + 1.0) = 1.0 / 1.0 = 1.0
    assert result.mean() == pytest.approx(1.0)


def test_fill_rate_mean_weighted_all_zero() -> None:
    """Test weighted mean when all weights are zero."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.5, fill_rate_weight=0.0)
        age: int = Spec(fill_rate_func=lambda x: 1.0, fill_rate_weight=0.0)

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    # When total weight is 0.0, mean should return 0.0
    assert result.mean() == 0.0


def test_fill_rate_mean_weighted_nested() -> None:
    """Test weighted mean with nested models."""

    class Address(BaseModel):
        street: str = Spec(fill_rate_func=lambda x: 0.5, fill_rate_weight=2.0)
        city: str = Spec(fill_rate_func=lambda x: 1.0, fill_rate_weight=1.0)

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.8, fill_rate_weight=1.0)
        address: Address

    person = Person.from_dict(
        {
            "name": "John",
            "address": {"street": "123 Main St", "city": "Anytown"},
        }
    )
    result = person.compute_fill_rate()

    # Weighted mean: (0.8 * 1.0 + 0.5 * 2.0 + 1.0 * 1.0) / (1.0 + 2.0 + 1.0)
    # = (0.8 + 1.0 + 1.0) / 4.0 = 2.8 / 4.0 = 0.7
    assert result.mean() == pytest.approx(0.7)


def test_fill_rate_mean_weighted_deeply_nested() -> None:
    """Test weighted mean with deeply nested models."""

    class Country(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.6, fill_rate_weight=1.5)
        code: str = Spec(fill_rate_func=lambda x: 1.0, fill_rate_weight=0.5)

    class Address(BaseModel):
        street: str = Spec(fill_rate_func=lambda x: 0.5, fill_rate_weight=2.0)
        country: Country

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.8, fill_rate_weight=1.0)
        address: Address

    person = Person.from_dict(
        {
            "name": "John",
            "address": {
                "street": "123 Main St",
                "country": {"name": "USA", "code": "US"},
            },
        }
    )
    result = person.compute_fill_rate()

    # Weighted mean: (0.8 * 1.0 + 0.5 * 2.0 + 0.6 * 1.5 + 1.0 * 0.5) /
    #                (1.0 + 2.0 + 1.5 + 0.5)
    # = (0.8 + 1.0 + 0.9 + 0.5) / 5.0 = 3.2 / 5.0 = 0.64
    assert result.mean() == pytest.approx(0.64)


def test_fill_rate_max_with_weights() -> None:
    """Test that max is unchanged with weights."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.3, fill_rate_weight=2.0)
        age: int = Spec(fill_rate_func=lambda x: 0.8, fill_rate_weight=0.5)
        email: str = Spec(fill_rate_func=lambda x: 0.5, fill_rate_weight=1.0)

    person = Person(name="John", age=30, email="john@example.com")
    result = person.compute_fill_rate()

    # Max should be 0.8 regardless of weights
    assert result.max() == 0.8


def test_fill_rate_min_with_weights() -> None:
    """Test that min is unchanged with weights."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.3, fill_rate_weight=2.0)
        age: int = Spec(fill_rate_func=lambda x: 0.8, fill_rate_weight=0.5)
        email: str = Spec(fill_rate_func=lambda x: 0.5, fill_rate_weight=1.0)

    person = Person(name="John", age=30, email="john@example.com")
    result = person.compute_fill_rate()

    # Min should be 0.3 regardless of weights
    assert result.min() == 0.3


def test_fill_rate_max_min_nested_with_weights() -> None:
    """Test max/min with nested models and weights."""

    class Address(BaseModel):
        street: str = Spec(fill_rate_func=lambda x: 0.2, fill_rate_weight=2.0)
        city: str = Spec(fill_rate_func=lambda x: 0.9, fill_rate_weight=1.0)

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.5, fill_rate_weight=1.0)
        address: Address

    person = Person.from_dict(
        {
            "name": "John",
            "address": {"street": "123 Main St", "city": "Anytown"},
        }
    )
    result = person.compute_fill_rate()

    # Max should be 0.9 (from city), min should be 0.2 (from street)
    assert result.max() == 0.9
    assert result.min() == 0.2


def test_fill_rate_weight_float_inf() -> None:
    """Test that weight can be +inf."""

    import math

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.5, fill_rate_weight=math.inf)
        age: int = Spec(fill_rate_func=lambda x: 1.0, fill_rate_weight=1.0)

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    # With inf weight, the weighted mean should approach the value with inf weight
    # (0.5 * inf + 1.0 * 1.0) / (inf + 1.0) ≈ 0.5
    assert result.fields.name.weight == math.inf
    # Mean calculation with inf should work (Python handles inf arithmetic)
    mean_value = result.mean()
    assert mean_value == pytest.approx(0.5, abs=0.1) or math.isnan(mean_value)


def test_fill_rate_weight_zero_sum() -> None:
    """Test weighted mean when sum of weights is zero but not all zero."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.5, fill_rate_weight=0.0)
        age: int = Spec(fill_rate_func=lambda x: 1.0, fill_rate_weight=0.0)

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    # When total weight is 0.0, mean should return 0.0
    assert result.mean() == 0.0


def test_fill_rate_weight_large_values() -> None:
    """Test weighted mean with very large weight values."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_func=lambda x: 0.5, fill_rate_weight=1e10)
        age: int = Spec(fill_rate_func=lambda x: 1.0, fill_rate_weight=1.0)

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    # Weighted mean should still work with large weights
    # (0.5 * 1e10 + 1.0 * 1.0) / (1e10 + 1.0) ≈ 0.5
    assert result.mean() == pytest.approx(0.5, abs=0.01)


def test_fill_rate_weight_negative_in_spec_raises_error() -> None:
    """Test that negative weight in Spec() raises InvalidWeightError."""

    with pytest.raises(InvalidWeightError) as exc_info:
        Spec(fill_rate_weight=-1.0)

    assert "Invalid weight in Spec" in str(exc_info.value)
    assert "-1.0" in str(exc_info.value)
    assert "must be >= 0.0" in str(exc_info.value)


def test_fill_rate_weight_negative_in_decorator_raises_error() -> None:
    """Test that negative weight in decorator raises InvalidWeightError."""

    with pytest.raises(InvalidWeightError) as exc_info:

        class Person(BaseModel):
            name: str

            @fill_rate_func("name", weight=-1.0)
            def fill_rate_name(x: t.Any) -> float:
                return 1.0

    assert "Invalid weight in decorator" in str(exc_info.value)
    assert "-1.0" in str(exc_info.value)
    assert "must be >= 0.0" in str(exc_info.value)


def test_fill_rate_weight_zero_allowed() -> None:
    """Test that weight = 0.0 is allowed."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_weight=0.0)
        age: int

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    assert result.fields.name.weight == 0.0
    assert result.fields.age.weight == 1.0


def test_fill_rate_weight_zero_allowed_in_decorator() -> None:
    """Test that weight = 0.0 is allowed in decorator."""

    class Person(BaseModel):
        name: str

        @fill_rate_func("name", weight=0.0)
        def fill_rate_name(x: t.Any) -> float:
            return 1.0 if x is not MissingValue else 0.0

    person = Person(name="John")
    result = person.compute_fill_rate()

    assert result.fields.name.weight == 0.0


def test_fill_rate_field_collection_resolve_path_basemodel_type() -> None:
    """Test resolving paths with BaseModel nested."""

    class Address(BaseModel):
        street: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict({"name": "John", "address": {"street": "123 Main"}})
    result = person.compute_fill_rate()

    # Access nested field in fill rate result
    street_result = result["address.street"]
    assert isinstance(street_result, FillRateFieldResult)
    assert street_result.value == 1.0


def test_fill_rate_field_collection_empty_segments_resolve() -> None:
    """Test resolving empty segments raises KeyError."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()

    # Empty segments should raise error
    with pytest.raises(KeyError, match="Empty path"):
        _ = result.fields._resolve_path([])


def test_fill_rate_list_primitive_missing() -> None:
    """Test that list[Primitive] with MissingValue returns 0.0."""

    class Person(BaseModel):
        name: str
        tags: list[str]

    person = Person(name="John")
    result = person.compute_fill_rate()

    assert result.fields.name.value == 1.0
    assert result.fields.tags.value == 0.0


def test_fill_rate_list_primitive_empty() -> None:
    """Test that empty list[Primitive] returns 0.0."""

    class Person(BaseModel):
        name: str
        tags: list[str]

    person = Person(name="John", tags=[])
    result = person.compute_fill_rate()

    assert result.fields.name.value == 1.0
    assert result.fields.tags.value == 0.0


def test_fill_rate_list_primitive_filled() -> None:
    """Test that non-empty list[Primitive] returns 1.0."""

    class Person(BaseModel):
        name: str
        tags: list[str]
        scores: list[int]

    person = Person(name="John", tags=["python", "rust"], scores=[85, 90])
    result = person.compute_fill_rate()

    assert result.fields.name.value == 1.0
    assert result.fields.tags.value == 1.0
    assert result.fields.scores.value == 1.0


def test_fill_rate_list_primitive_custom_func() -> None:
    """Test that custom fill_rate_func works with list[Primitive]."""

    class Person(BaseModel):
        tags: list[str] = Spec(fill_rate_func=lambda x: len(x) / 10)

    person = Person(tags=["a", "b", "c"])
    result = person.compute_fill_rate()

    assert result.fields.tags.value == 0.3  # 3/10


def test_fill_rate_list_primitive_with_weight() -> None:
    """Test that weight works with list[Primitive]."""

    class Person(BaseModel):
        name: str
        tags: list[str] = Spec(fill_rate_weight=2.0)

    person = Person(name="John", tags=["python"])
    result = person.compute_fill_rate()

    assert result.fields.name.weight == 1.0
    assert result.fields.tags.weight == 2.0
    assert result.fields.tags.value == 1.0


def test_fill_rate_list_basemodel_returns_list_result() -> None:
    """Test that list[BaseModel] returns FillRateListResult."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        customer: str
        items: list[Item]

    order = Order.from_dict(
        {
            "customer": "John",
            "items": [{"name": "Apple", "price": 1.0}],
        }
    )
    result = order.compute_fill_rate()

    assert isinstance(result.fields.items, FillRateListResult)
    assert len(result.fields.items) == 1


def test_fill_rate_list_basemodel_getitem() -> None:
    """Test that FillRateListResult supports __getitem__."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana", "price": 0.5},
            ],
        }
    )
    result = order.compute_fill_rate()

    assert isinstance(result.fields.items[0], FillRateModelResult)
    assert isinstance(result.fields.items[1], FillRateModelResult)
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0


def test_fill_rate_list_basemodel_len() -> None:
    """Test that FillRateListResult supports len()."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
                {"name": "Cherry"},
            ],
        }
    )
    result = order.compute_fill_rate()

    assert len(result.fields.items) == 3


def test_fill_rate_list_basemodel_iter() -> None:
    """Test that FillRateListResult supports iteration."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
            ],
        }
    )
    result = order.compute_fill_rate()

    items_list = list(result.fields.items)
    assert len(items_list) == 2
    assert all(isinstance(item, FillRateModelResult) for item in items_list)


def test_fill_rate_list_basemodel_empty() -> None:
    """Test that empty list[BaseModel] returns FillRateListResult with empty items."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict({"items": []})
    result = order.compute_fill_rate()

    assert isinstance(result.fields.items, FillRateListResult)
    assert len(result.fields.items) == 0
    assert result.fields.items.mean() == 0.0


def test_fill_rate_list_basemodel_missing() -> None:
    """Test that MissingValue list[BaseModel] returns FillRateListResult with empty items."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        customer: str
        items: list[Item]

    order = Order(customer="John")
    result = order.compute_fill_rate()

    assert isinstance(result.fields.items, FillRateListResult)
    assert len(result.fields.items) == 0
    assert result.fields.items.mean() == 0.0


def test_fill_rate_list_basemodel_item_access() -> None:
    """Test accessing individual items from FillRateListResult."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana"},  # price missing
            ],
        }
    )
    result = order.compute_fill_rate()

    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0
    assert result.fields.items[1].fields.name.value == 1.0
    assert result.fields.items[1].fields.price.value == 0.0


def test_fill_rate_list_basemodel_mean() -> None:
    """Test that FillRateListResult.mean() calculates mean across all items."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana"},  # price missing
            ],
        }
    )
    result = order.compute_fill_rate()

    # item 0: name=1.0, price=1.0 -> mean=1.0
    # item 1: name=1.0, price=0.0 -> mean=0.5
    # Global mean: (1.0 + 0.5) / 2 = 0.75
    assert result.fields.items.mean() == pytest.approx(0.75)


def test_fill_rate_list_aggregated_field_access() -> None:
    """Test that aggregated field access works: items.name."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana", "price": 0.5},
            ],
        }
    )
    result = order.compute_fill_rate()

    assert isinstance(
        result.fields.items.aggregated_fields.name, FillRateAggregatedFieldResult
    )
    assert isinstance(
        result.fields.items.aggregated_fields.price, FillRateAggregatedFieldResult
    )

    # Test aggregated_fields API (recommended)
    assert isinstance(
        result.fields.items.aggregated_fields.name, FillRateAggregatedFieldResult
    )
    assert isinstance(
        result.fields.items.aggregated_fields.price, FillRateAggregatedFieldResult
    )


def test_fill_rate_list_aggregated_values() -> None:
    """Test that aggregated field provides values list."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana"},  # price missing
                {"price": 2.0},  # name missing
            ],
        }
    )
    result = order.compute_fill_rate()

    assert result.fields.items.aggregated_fields.name.values == [1.0, 1.0, 0.0]
    assert result.fields.items.aggregated_fields.price.values == [1.0, 0.0, 1.0]


def test_fill_rate_list_aggregated_mean() -> None:
    """Test that aggregated field mean() works."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
                {},  # name missing
            ],
        }
    )
    result = order.compute_fill_rate()

    # name values: [1.0, 1.0, 0.0] -> mean = 2/3
    assert result.fields.items.aggregated_fields.name.mean() == pytest.approx(2.0 / 3.0)


def test_fill_rate_list_aggregated_max_min() -> None:
    """Test that aggregated field max() and min() work."""

    class Item(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"score": 50.0},
                {"score": 80.0},
                {"score": 30.0},
            ],
        }
    )
    result = order.compute_fill_rate()

    assert result.fields.items.aggregated_fields.score.max() == 0.8
    assert result.fields.items.aggregated_fields.score.min() == 0.3


def test_fill_rate_list_aggregated_std_var() -> None:
    """Test that aggregated field std() and var() work."""

    class Item(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"score": 50.0},
                {"score": 80.0},
                {"score": 30.0},
            ],
        }
    )
    result = order.compute_fill_rate()

    values = [0.5, 0.8, 0.3]
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    std_val = variance**0.5

    assert result.fields.items.aggregated_fields.score.std() == pytest.approx(std_val)
    assert result.fields.items.aggregated_fields.score.var() == pytest.approx(variance)


def test_fill_rate_list_aggregated_quantile() -> None:
    """Test that aggregated field quantile() works."""

    class Item(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"score": 30.0},
                {"score": 50.0},
                {"score": 80.0},
            ],
        }
    )
    result = order.compute_fill_rate()

    assert result.fields.items.aggregated_fields.score.quantile(0.5) == pytest.approx(
        0.5
    )


def test_fill_rate_list_aggregated_empty() -> None:
    """Test that aggregated field works with empty list."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict({"items": []})
    result = order.compute_fill_rate()

    assert result.fields.items.aggregated_fields.name.values == []
    assert result.fields.items.aggregated_fields.name.mean() == 0.0
    assert result.fields.items.aggregated_fields.name.max() == 0.0
    assert result.fields.items.aggregated_fields.name.min() == 0.0


def test_fill_rate_list_aggregated_nested_model() -> None:
    """Test that aggregated access works for nested models: items.address.city."""

    class Address(BaseModel):
        city: str
        street: str

    class Item(BaseModel):
        name: str
        address: Address

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "address": {"city": "NYC", "street": "Main"}},
                {"name": "Banana", "address": {"city": "LA"}},  # street missing
            ],
        }
    )
    result = order.compute_fill_rate()

    assert isinstance(
        result.fields.items.aggregated_fields.address, FillRateAggregatedModelResult
    )
    assert isinstance(
        result.fields.items.aggregated_fields.address.city,
        FillRateAggregatedFieldResult,
    )
    assert result.fields.items.aggregated_fields.address.city.values == [1.0, 1.0]
    assert result.fields.items.aggregated_fields.address.street.values == [1.0, 0.0]

    # Test aggregated_fields API (recommended)
    assert isinstance(
        result.fields.items.aggregated_fields.address, FillRateAggregatedModelResult
    )
    # FillRateAggregatedModelResult uses __getattr__ directly, not aggregated_fields
    assert isinstance(
        result.fields.items.aggregated_fields.address.city,
        FillRateAggregatedFieldResult,
    )
    assert result.fields.items.aggregated_fields.address.city.values == [1.0, 1.0]
    assert result.fields.items.aggregated_fields.address.street.values == [1.0, 0.0]


def test_fill_rate_list_aggregated_zero_weight() -> None:
    """Test that aggregated field with zero weight returns 0.0 mean."""

    class Item(BaseModel):
        name: str = Spec(fill_rate_weight=0.0)

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
            ],
        }
    )
    result = order.compute_fill_rate()

    # All items have weight 0.0, so mean should be 0.0
    assert result.fields.items.aggregated_fields.name.mean() == 0.0


def test_fill_rate_list_aggregated_single_value() -> None:
    """Test that aggregated field with single value has std=0 and var=0."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
            ],
        }
    )
    result = order.compute_fill_rate()

    assert result.fields.items.aggregated_fields.name.std() == 0.0
    assert result.fields.items.aggregated_fields.name.var() == 0.0


def test_fill_rate_list_aggregated_quantile_invalid() -> None:
    """Test that aggregated field quantile raises ValueError for invalid q."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
            ],
        }
    )
    result = order.compute_fill_rate()

    with pytest.raises(ValueError, match="Quantile q must be between 0.0 and 1.0"):
        _ = result.fields.items.aggregated_fields.name.quantile(1.5)

    with pytest.raises(ValueError, match="Quantile q must be between 0.0 and 1.0"):
        _ = result.fields.items.aggregated_fields.name.quantile(-0.1)


def test_fill_rate_list_aggregated_quantile_empty() -> None:
    """Test that aggregated field quantile returns 0.0 for empty values."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict({"items": []})
    result = order.compute_fill_rate()

    assert result.fields.items.aggregated_fields.name.quantile(0.5) == 0.0


def test_fill_rate_list_aggregated_quantile_boundaries() -> None:
    """Test that aggregated field quantile works at boundaries 0.0 and 1.0."""

    class Item(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"score": 30.0},
                {"score": 50.0},
                {"score": 80.0},
            ],
        }
    )
    result = order.compute_fill_rate()

    # quantile(0.0) should return min value
    assert result.fields.items.aggregated_fields.score.quantile(0.0) == 0.3
    # quantile(1.0) should return max value
    assert result.fields.items.aggregated_fields.score.quantile(1.0) == 0.8


def test_fill_rate_list_aggregated_nested_list() -> None:
    """Test that aggregated access works for nested lists: items.subitems returns mean."""

    class SubItem(BaseModel):
        name: str

    class Item(BaseModel):
        name: str
        subitems: list[SubItem]

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {
                    "name": "Item1",
                    "subitems": [
                        {"name": "Sub1"},
                        {"name": "Sub2"},
                    ],
                },
                {
                    "name": "Item2",
                    "subitems": [
                        {"name": "Sub3"},
                    ],
                },
            ],
        }
    )
    result = order.compute_fill_rate()

    # Access nested list aggregated - returns FillRateNestedListAggregatedResult
    assert isinstance(
        result.fields.items.aggregated_fields.subitems,
        FillRateNestedListAggregatedResult,
    )
    # Each item's subitems list has mean=1.0 (all fields filled)
    # So values should be [1.0, 1.0] (one value per item)
    assert result.fields.items.aggregated_fields.subitems.values == [1.0, 1.0]


def test_fill_rate_list_aggregated_model_empty() -> None:
    """Test that aggregated access works with empty list."""

    class Address(BaseModel):
        city: str

    class Item(BaseModel):
        name: str
        address: Address

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict({"items": []})
    result = order.compute_fill_rate()

    # Empty list should return empty aggregated field result
    assert isinstance(
        result.fields.items.aggregated_fields.address, FillRateAggregatedFieldResult
    )
    assert result.fields.items.aggregated_fields.address.values == []
    assert result.fields.items.aggregated_fields.address.mean() == 0.0
    assert result.fields.items.aggregated_fields.address.max() == 0.0
    assert result.fields.items.aggregated_fields.address.min() == 0.0


def test_fill_rate_list_aggregated_model_stats() -> None:
    """Test that FillRateAggregatedModelResult stats work."""

    class Address(BaseModel):
        city: str = Spec(fill_rate_func=lambda x: 0.5 if x == "NYC" else 1.0)

    class Item(BaseModel):
        name: str
        address: Address

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Item1", "address": {"city": "NYC"}},
                {"name": "Item2", "address": {"city": "LA"}},
                {"name": "Item3", "address": {"city": "SF"}},
            ],
        }
    )
    result = order.compute_fill_rate()

    # address.city values: [0.5, 1.0, 1.0]
    # address mean per item: [0.5, 1.0, 1.0]
    assert isinstance(
        result.fields.items.aggregated_fields.address, FillRateAggregatedModelResult
    )
    assert result.fields.items.aggregated_fields.address.mean() == pytest.approx(
        0.8333, abs=0.01
    )
    assert result.fields.items.aggregated_fields.address.max() == 1.0
    assert result.fields.items.aggregated_fields.address.min() == 0.5


def test_fill_rate_list_aggregated_model_std_var() -> None:
    """Test that FillRateAggregatedModelResult std and var work."""

    class Address(BaseModel):
        city: str = Spec(fill_rate_func=lambda x: 0.3 if x == "A" else 0.8)

    class Item(BaseModel):
        address: Address

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"address": {"city": "A"}},
                {"address": {"city": "B"}},
                {"address": {"city": "B"}},
            ],
        }
    )
    result = order.compute_fill_rate()

    # address mean per item: [0.3, 0.8, 0.8]
    assert isinstance(
        result.fields.items.aggregated_fields.address, FillRateAggregatedModelResult
    )
    means = [0.3, 0.8, 0.8]
    mean_val = sum(means) / len(means)
    variance = sum((x - mean_val) ** 2 for x in means) / len(means)
    std_val = variance**0.5

    assert result.fields.items.aggregated_fields.address.std() == pytest.approx(std_val)
    assert result.fields.items.aggregated_fields.address.var() == pytest.approx(
        variance
    )


def test_fill_rate_list_aggregated_model_single_item() -> None:
    """Test that FillRateAggregatedModelResult works with single item."""

    class Address(BaseModel):
        city: str

    class Item(BaseModel):
        name: str
        address: Address

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Item1", "address": {"city": "NYC"}},
            ],
        }
    )
    result = order.compute_fill_rate()

    # Single item should work
    assert isinstance(
        result.fields.items.aggregated_fields.address, FillRateAggregatedModelResult
    )
    assert result.fields.items.aggregated_fields.address.mean() == 1.0
    assert (
        result.fields.items.aggregated_fields.address.std() == 0.0
    )  # Single item -> std = 0
    assert (
        result.fields.items.aggregated_fields.address.var() == 0.0
    )  # Single item -> var = 0


def test_fill_rate_list_result_value_property() -> None:
    """Test that FillRateListResult.value property works."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
            ],
        }
    )
    result = order.compute_fill_rate()

    # value property should return mean()
    assert result.fields.items.value == result.fields.items.mean()


def test_fill_rate_list_result_empty_all_values() -> None:
    """Test FillRateListResult when all items have no values."""

    class EmptyItem(BaseModel):
        pass

    class Order(BaseModel):
        items: list[EmptyItem]

    order = Order.from_dict(
        {
            "items": [
                {},
                {},
            ],
        }
    )
    result = order.compute_fill_rate()

    # Empty items should return mean=0.0
    assert result.fields.items.mean() == 0.0


def test_fill_rate_list_result_zero_weight() -> None:
    """Test FillRateListResult when all weights are zero."""

    class Item(BaseModel):
        name: str = Spec(fill_rate_weight=0.0)

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
            ],
        }
    )
    result = order.compute_fill_rate()

    # All items have weight 0.0, so mean should be 0.0
    assert result.fields.items.mean() == 0.0


def test_fill_rate_list_aggregated_nested_model_access() -> None:
    """Test that accessing nested model through aggregated works."""

    class Address(BaseModel):
        city: str
        street: str

    class Item(BaseModel):
        name: str
        address: Address

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Item1", "address": {"city": "NYC", "street": "Main"}},
                {"name": "Item2", "address": {"city": "LA", "street": "Oak"}},
            ],
        }
    )
    result = order.compute_fill_rate()

    # Access nested model through aggregated
    assert isinstance(
        result.fields.items.aggregated_fields.address, FillRateAggregatedModelResult
    )
    # Then access field of nested model
    assert isinstance(
        result.fields.items.aggregated_fields.address.city,
        FillRateAggregatedFieldResult,
    )
    assert result.fields.items.aggregated_fields.address.city.values == [1.0, 1.0]


def test_fill_rate_aggregated_model_empty_items_direct() -> None:
    """Test FillRateAggregatedModelResult with empty _items list directly."""

    from cobjectric.fill_rate import FillRateAggregatedModelResult

    empty_aggregated = FillRateAggregatedModelResult(_items=[])
    assert empty_aggregated.mean() == 0.0
    assert empty_aggregated.max() == 0.0
    assert empty_aggregated.min() == 0.0


def test_fill_rate_model_result_with_list_collects_values() -> None:
    """Test that FillRateModelResult._collect_all_values includes FillRateListResult."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        customer: str
        items: list[Item]

    order = Order.from_dict(
        {
            "customer": "John",
            "items": [
                {"name": "Apple"},
                {"name": "Banana"},
            ],
        }
    )
    result = order.compute_fill_rate()

    # _collect_all_values should include values from list items
    all_values = result._collect_all_values()
    # customer=1.0, items[0].name=1.0, items[1].name=1.0
    assert len(all_values) == 3
    assert all(v == 1.0 for v in all_values)


def test_fill_rate_model_result_with_list_collects_values_and_weights() -> None:
    """Test that FillRateModelResult._collect_all_values_and_weights includes FillRateListResult."""

    class Item(BaseModel):
        name: str = Spec(fill_rate_weight=2.0)

    class Order(BaseModel):
        customer: str
        items: list[Item]

    order = Order.from_dict(
        {
            "customer": "John",
            "items": [
                {"name": "Apple"},
            ],
        }
    )
    result = order.compute_fill_rate()

    # _collect_all_values_and_weights should include values and weights from list items
    all_values, all_weights = result._collect_all_values_and_weights()
    # customer=1.0 weight=1.0, items[0].name=1.0 weight=2.0
    assert len(all_values) == 2
    assert len(all_weights) == 2
    assert all_values == [1.0, 1.0]
    assert all_weights == [1.0, 2.0]


def test_fill_rate_list_aggregated_nested_list_explicit() -> None:
    """Test that accessing nested list through aggregated works."""

    class SubItem(BaseModel):
        name: str

    class Item(BaseModel):
        name: str
        subitems: list[SubItem]

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {
                    "name": "Item1",
                    "subitems": [
                        {"name": "Sub1"},
                    ],
                },
                {
                    "name": "Item2",
                    "subitems": [
                        {"name": "Sub2"},
                    ],
                },
            ],
        }
    )
    result = order.compute_fill_rate()

    # Access nested list through aggregated - should return FillRateNestedListAggregatedResult
    assert isinstance(
        result.fields.items.aggregated_fields.subitems,
        FillRateNestedListAggregatedResult,
    )
    # Each item's subitems list has mean=1.0, so values should be [1.0, 1.0]
    assert result.fields.items.aggregated_fields.subitems.values == [1.0, 1.0]


def test_field_collection_list_access_basemodel():
    """Test path access on BaseModel list field with [index] notation."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana", "price": 0.5},
            ],
        }
    )

    # Test index access on BaseModel list
    item0 = order["items[0]"]  # Should return the Item instance
    assert isinstance(item0, BaseModel)
    assert item0.fields.name.value == "Apple"

    # Test nested field access through list index
    name0 = order["items[0].name"]
    assert isinstance(name0, Field)
    assert name0.value == "Apple"

    price1 = order["items[1].price"]
    assert isinstance(price1, Field)
    assert price1.value == 0.5


def test_field_collection_list_access_primitive():
    """Test path access on primitive list field with [index] notation."""

    class Person(BaseModel):
        tags: list[str]

    person = Person.from_dict({"tags": ["python", "rust", "go"]})

    # Test index access on primitive list
    tag0 = person["tags[0]"]
    assert tag0 == "python"

    tag1 = person["tags[1]"]
    assert tag1 == "rust"

    tag2 = person["tags[2]"]
    assert tag2 == "go"


def test_field_collection_list_access_deeply_nested():
    """Test deeply nested path access with multiple list indices."""

    class Address(BaseModel):
        city: str

    class Item(BaseModel):
        name: str
        address: Address

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Item1", "address": {"city": "NYC"}},
                {"name": "Item2", "address": {"city": "LA"}},
            ],
        }
    )

    # Test deeply nested access: items[0].address.city
    city0 = order["items[0].address.city"]
    assert isinstance(city0, Field)
    assert city0.value == "NYC"

    city1 = order["items[1].address.city"]
    assert isinstance(city1, Field)
    assert city1.value == "LA"


def test_field_collection_nested_primitive_list_access():
    """Test path access on nested list[list[str]] with [index] notation."""

    class Container(BaseModel):
        lists: list[list[str]]

    container = Container.from_dict(
        {
            "lists": [
                ["a", "b", "c"],
                ["x", "y", "z"],
            ],
        }
    )

    # Test index access on nested primitive list
    # This exercises the isinstance(current, list) branch in _resolve_path
    first_list = container["lists[0]"]
    assert isinstance(first_list, list)
    assert first_list == ["a", "b", "c"]

    # Access element within nested list
    element = container["lists[0][1]"]
    assert element == "b"

    element2 = container["lists[1][0]"]
    assert element2 == "x"


def test_field_collection_list_out_of_range():
    """Test IndexError when accessing out of range list index."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Item1"},
            ],
        }
    )

    # Test out of range access
    with pytest.raises(KeyError, match="List index .* out of range"):
        _ = order["items[5]"]


def test_field_collection_nested_list_out_of_range():
    """Test IndexError on nested list with out of range index."""

    class Container(BaseModel):
        lists: list[list[str]]

    container = Container.from_dict(
        {
            "lists": [
                ["a", "b"],
            ],
        }
    )

    # Test out of range access on nested list
    # This exercises the IndexError except block for isinstance(current, list)
    with pytest.raises(KeyError, match="List index .* out of range"):
        _ = container["lists[0][5]"]


def test_field_collection_basemodel_in_list_updates_fields():
    """Test that accessing a BaseModel in a list updates current_fields."""

    class Address(BaseModel):
        city: str
        street: str

    class Item(BaseModel):
        address: Address

    class Container(BaseModel):
        items: list[Item]

    container = Container.from_dict(
        {
            "items": [
                {
                    "address": {
                        "city": "NYC",
                        "street": "5th Ave",
                    }
                },
            ],
        }
    )

    # Access nested Address through list index
    # This tests the isinstance(current, BaseModel) branch that updates current_fields
    address_city = container["items[0].address.city"]
    assert address_city.value == "NYC"

    address_street = container["items[0].address.street"]
    assert address_street.value == "5th Ave"


def test_field_collection_cannot_index_non_list_field():
    """Test KeyError when trying to index a non-list field."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Item1"},
            ],
        }
    )

    # Try to access index on a string field (should fail)
    # This exercises the "Cannot use index on non-list field" branch for Field
    error_raised = False
    try:
        _ = order["items[0].name[0]"]
    except KeyError as e:
        error_raised = True
        assert "Cannot use index on non-list field" in str(e)
    assert error_raised


def test_fill_rate_list_nested_list_aggregation():
    """Test aggregation of nested lists in FillRateListResult."""

    class SubItem(BaseModel):
        value: str

    class Item(BaseModel):
        subitems: list[SubItem]

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"subitems": [{"value": "a"}, {"value": "b"}]},
                {"subitems": [{"value": "c"}]},
            ],
        }
    )

    result = order.compute_fill_rate()

    # Access aggregated nested list
    subitems_agg = result.fields.items.aggregated_fields.subitems
    assert isinstance(subitems_agg, FillRateNestedListAggregatedResult)
    # First item has 2 subitems (mean=1.0), second has 1 (mean=1.0)
    assert subitems_agg.values == [1.0, 1.0]


def test_field_collection_basemodel_list_updates_fields_via_path():
    """Test that accessing a BaseModel in a list via path updates current_fields."""

    class Address(BaseModel):
        city: str

    class Item(BaseModel):
        address: Address

    class Container(BaseModel):
        items: list[Item]

    container = Container.from_dict(
        {
            "items": [
                {"address": {"city": "NYC"}},
                {"address": {"city": "LA"}},
            ],
        }
    )

    # Accessing items[0] returns an Item instance, then we access its address.city
    city = container["items[0].address.city"]
    assert isinstance(city, Field)
    assert city.value == "NYC"

    # Access second item to exercise the path resolution again
    city2 = container["items[1].address.city"]
    assert city2.value == "LA"


def test_fill_rate_list_basemodel_nested_aggregation_complete():
    """Test complete aggregation path for nested models in lists."""

    class SubItem(BaseModel):
        name: str

    class Item(BaseModel):
        subitems: list[SubItem]

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"subitems": [{"name": "a"}, {"name": "b"}]},
                {"subitems": [{"name": "c"}]},
                {"subitems": [{"name": "d"}, {"name": "e"}, {"name": "f"}]},
            ],
        }
    )

    result = order.compute_fill_rate()

    # Access aggregated nested lists to exercise all paths
    subitems_agg = result.fields.items.aggregated_fields.subitems
    assert isinstance(subitems_agg, FillRateNestedListAggregatedResult)
    assert subitems_agg.values == [1.0, 1.0, 1.0]
    assert subitems_agg.mean() == 1.0
    assert subitems_agg.min() == 1.0
    assert subitems_agg.max() == 1.0


def test_fill_rate_list_aggregated_nested_model_returns_aggregated_model_result():
    """Test that accessing a nested model field via aggregated access
    returns FillRateAggregatedModelResult.

    When accessing a nested model field through aggregated_fields,
    it should return FillRateAggregatedModelResult containing all
    nested model results.
    """

    class Address(BaseModel):
        city: str
        street: str

    class Item(BaseModel):
        name: str
        address: Address  # This is a nested model, not a field or list

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Item1", "address": {"city": "NYC", "street": "5th Ave"}},
                {"name": "Item2", "address": {"city": "LA", "street": "Sunset Blvd"}},
            ],
        }
    )

    result = order.compute_fill_rate()

    # Access the nested model via aggregated access
    # This should return FillRateAggregatedModelResult
    address_agg = result.fields.items.aggregated_fields.address
    assert isinstance(address_agg, FillRateAggregatedModelResult)

    # Access nested field within the aggregated model result
    city_agg = address_agg.city
    assert isinstance(city_agg, FillRateAggregatedFieldResult)
    assert city_agg.values == [1.0, 1.0]


def test_field_collection_index_on_non_list_field_direct():
    """Test that accessing an index on a non-list Field raises KeyError."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")

    # Accessing name[0] should fail because name is a string, not a list
    # The path is parsed as ["name", "[0]"]
    # When we try to access [0] on the Field(name), it should raise
    raised = False
    try:
        _ = person["name[0]"]
    except KeyError as e:
        raised = True
        assert "Cannot use index on non-list field" in str(e)
    assert raised, "Expected KeyError for indexing non-list field"


def test_field_collection_list_of_basemodel_updates_current_fields():
    """Test that accessing a BaseModel from a list updates current_fields."""

    class Address(BaseModel):
        city: str

    class Item(BaseModel):
        address: Address

    class Container(BaseModel):
        data: list[list[Item]]

    container = Container.from_dict(
        {
            "data": [
                [{"address": {"city": "NYC"}}],
                [{"address": {"city": "LA"}}],
            ],
        }
    )

    # Access data[0][0].address.city
    # After data[0] we get a list, then [0] gives us an Item BaseModel
    # current_fields should be updated when accessing the BaseModel
    city = container["data[0][0].address.city"]
    assert city.value == "NYC"


def test_field_collection_list_followed_by_index():
    """Test path access where current is a list followed by another index."""

    class Container(BaseModel):
        data: list[list[str]]

    container = Container.from_dict(
        {
            "data": [
                ["a", "b", "c"],
                ["x", "y", "z"],
            ],
        }
    )

    # Access data[0][1] - after data[0] we have a list, then [1] accesses element
    element = container["data[0][1]"]
    assert element == "b"

    element2 = container["data[1][2]"]
    assert element2 == "z"


def test_field_collection_cannot_index_on_primitive_in_list():
    """Test that trying to index a primitive value from a list raises KeyError."""

    class Container(BaseModel):
        data: list[list[str]]

    container = Container.from_dict(
        {
            "data": [
                ["hello", "world"],
            ],
        }
    )

    # Access data[0][0][0] - data[0][0] is "hello" (a string), not a list
    # Trying to index [0] on a string should fail
    raised = False
    try:
        _ = container["data[0][0][0]"]
    except KeyError as e:
        raised = True
        assert "Cannot use index on non-list element" in str(e)
    assert raised, "Expected KeyError for indexing string"


def test_fill_rate_aggregated_nested_model_via_list():
    """Test FillRateListAggregatedProxy.__getattr__ returns FillRateAggregatedModelResult."""

    class Address(BaseModel):
        city: str
        zip_code: str

    class Person(BaseModel):
        name: str
        address: Address

    class Company(BaseModel):
        employees: list[Person]

    company = Company.from_dict(
        {
            "employees": [
                {"name": "Alice", "address": {"city": "NYC", "zip_code": "10001"}},
                {"name": "Bob", "address": {"city": "LA", "zip_code": "90001"}},
                {"name": "Charlie", "address": {"city": "SF", "zip_code": "94101"}},
            ],
        }
    )

    result = company.compute_fill_rate()

    # Access nested model field through aggregated access
    # employees.address should return FillRateAggregatedModelResult
    employees_list = result.fields.employees
    # Use aggregated_fields access
    address_agg = employees_list.aggregated_fields.address
    assert isinstance(address_agg, FillRateAggregatedModelResult)

    # Access field within aggregated model result
    city_agg = address_agg.city
    assert isinstance(city_agg, FillRateAggregatedFieldResult)
    assert city_agg.values == [1.0, 1.0, 1.0]
    assert city_agg.mean() == 1.0


def test_fill_rate_aggregated_nested_list_in_list():
    """Test FillRateListAggregatedProxy.__getattr__ with nested list returns aggregated values."""

    class Tag(BaseModel):
        name: str

    class Item(BaseModel):
        title: str
        tags: list[Tag]

    class Catalog(BaseModel):
        items: list[Item]

    catalog = Catalog.from_dict(
        {
            "items": [
                {"title": "Item1", "tags": [{"name": "a"}, {"name": "b"}]},
                {"title": "Item2", "tags": [{"name": "c"}]},
            ],
        }
    )

    result = catalog.compute_fill_rate()

    # Access nested list through aggregated access
    # items.tags should return FillRateNestedListAggregatedResult with mean values
    items_list = result.fields.items
    # Use aggregated_fields access
    tags_agg = items_list.aggregated_fields.tags
    assert isinstance(tags_agg, FillRateNestedListAggregatedResult)
    # Each item's tags list has mean fill rate of 1.0
    assert tags_agg.values == [1.0, 1.0]


def test_repr_aggregated_field_collection() -> None:
    """Test __repr__ of FillRateAggregatedFieldCollection."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana", "price": 0.5},
            ],
        }
    )
    result = order.compute_fill_rate()

    repr_str = repr(result.fields.items.aggregated_fields)
    assert "FillRateAggregatedFieldCollection" in repr_str
    assert "name=..." in repr_str
    assert "price=..." in repr_str


def test_invalid_aggregated_field_error() -> None:
    """Test that accessing non-existent field raises InvalidAggregatedFieldError."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
            ],
        }
    )
    result = order.compute_fill_rate()

    # Try to access non-existent field
    with pytest.raises(InvalidAggregatedFieldError) as exc_info:
        _ = result.fields.items.aggregated_fields.non_existent

    assert "non_existent" in str(exc_info.value)
    assert "name" in str(exc_info.value)
    assert "price" in str(exc_info.value)


def test_nested_list_aggregated_result() -> None:
    """Test FillRateNestedListAggregatedResult for nested lists."""

    class A(BaseModel):
        name: str

    class B(BaseModel):
        description: str
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                {"description": "B1", "items": [{"name": "A1"}, {"name": "A2"}]},
                {"description": "B2", "items": [{"name": "A3"}]},
            ],
        }
    )
    result = c.compute_fill_rate()

    # Access nested list through aggregated
    nested_agg = result.fields.b_list.aggregated_fields.items
    assert isinstance(nested_agg, FillRateNestedListAggregatedResult)

    # Access fields in nested list
    names_agg = nested_agg.aggregated_fields.name
    assert isinstance(names_agg, FillRateAggregatedFieldResult)
    assert names_agg.values == [1.0, 1.0, 1.0]


def test_nested_list_aggregated_hierarchical_mean() -> None:
    """Test hierarchical vs flatten mean for nested lists."""

    class A(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class B(BaseModel):
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                # First B: items with scores 0.5, 1.0 -> mean = 0.75
                {"items": [{"score": 50.0}, {"score": 100.0}]},
                # Second B: items with score 0.3 -> mean = 0.3
                {"items": [{"score": 30.0}]},
            ],
        }
    )
    result = c.compute_fill_rate()

    nested_agg = result.fields.b_list.aggregated_fields.items
    scores_agg = nested_agg.aggregated_fields.score

    # Flatten: all values are [0.5, 1.0, 0.3]
    assert scores_agg.values == [0.5, 1.0, 0.3]
    assert scores_agg.mean() == pytest.approx((0.5 + 1.0 + 0.3) / 3)

    # Hierarchical: means are [0.75, 0.3]
    assert nested_agg.mean(hierarchical=True) == pytest.approx((0.75 + 0.3) / 2)


def test_nested_list_aggregated_hierarchical_std() -> None:
    """Test hierarchical std for nested lists."""

    class A(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class B(BaseModel):
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                {"items": [{"score": 0.0}, {"score": 100.0}]},  # mean = 0.5
                {"items": [{"score": 100.0}]},  # mean = 1.0
            ],
        }
    )
    result = c.compute_fill_rate()

    nested_agg = result.fields.b_list.aggregated_fields.items

    # Hierarchical std: means are [0.5, 1.0], mean of means = 0.75
    # std = sqrt(((0.5-0.75)^2 + (1.0-0.75)^2) / 2) = sqrt(0.0625) = 0.25
    assert nested_agg.std(hierarchical=True) == pytest.approx(0.25)


def test_deeply_nested_aggregated_fields() -> None:
    """Test accessing fields from the example in requirements."""

    class A(BaseModel):
        name: str

    class B(BaseModel):
        items_list: list[A]

    class C(BaseModel):
        b: B
        lb: list[B]

    c = C.from_dict(
        {
            "b": {"items_list": [{"name": "test"}]},
            "lb": [{"items_list": [{"name": "test2"}, {"name": "test3"}]}],
        }
    )
    result = c.compute_fill_rate()

    # Single nested access
    assert result.fields.b.fields.items_list.aggregated_fields.name.values == [1.0]

    # Deeply nested access
    assert isinstance(
        result.fields.lb.aggregated_fields.items_list,
        FillRateNestedListAggregatedResult,
    )
    names = result.fields.lb.aggregated_fields.items_list.aggregated_fields.name
    assert names.values == [1.0, 1.0]


def test_repr_field_collection_elided() -> None:
    """Test that FieldCollection repr is elided."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person = Person(name="John", age=30, email="john@example.com")

    repr_str = repr(person.fields)
    assert "FieldCollection" in repr_str
    assert "name=..." in repr_str
    assert "age=..." in repr_str
    assert "email=..." in repr_str
    # Should not contain the actual values
    assert "John" not in repr_str
    assert "30" not in repr_str


def test_fill_rate_field_collection_repr_elided() -> None:
    """Test that FillRateFieldCollection repr is elided."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John", age=30)
    result = person.compute_fill_rate()

    repr_str = repr(result.fields)
    assert "FillRateFieldCollection" in repr_str
    assert "name=..." in repr_str
    assert "age=..." in repr_str
    # Should not contain actual fill rate values
    assert "value=" not in repr_str
    assert "weight=" not in repr_str


def test_nested_list_aggregated_empty() -> None:
    """Test FillRateNestedListAggregatedResult with empty lists."""

    class A(BaseModel):
        name: str

    class B(BaseModel):
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                {"items": []},
                {"items": []},
            ],
        }
    )
    result = c.compute_fill_rate()

    nested_agg = result.fields.b_list.aggregated_fields.items
    assert nested_agg.mean() == 0.0
    assert nested_agg.std() == 0.0
    assert nested_agg.min() == 0.0
    assert nested_agg.max() == 0.0


def test_nested_list_aggregated_quantile() -> None:
    """Test quantile for nested list aggregation."""

    class A(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class B(BaseModel):
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                {"items": [{"score": 30.0}]},  # mean = 0.3
                {"items": [{"score": 50.0}, {"score": 70.0}]},  # mean = 0.6
                {"items": [{"score": 100.0}]},  # mean = 1.0
            ],
        }
    )
    result = c.compute_fill_rate()

    nested_agg = result.fields.b_list.aggregated_fields.items

    # Hierarchical: means are [0.3, 0.6, 1.0]
    assert nested_agg.quantile(0.5, hierarchical=True) == pytest.approx(0.6)
    assert nested_agg.quantile(0.0, hierarchical=True) == pytest.approx(0.3)
    assert nested_agg.quantile(1.0, hierarchical=True) == pytest.approx(1.0)


def test_nested_list_aggregated_repr() -> None:
    """Test __repr__ of FillRateNestedListAggregatedResult."""

    class A(BaseModel):
        name: str

    class B(BaseModel):
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                {"items": [{"name": "A1"}, {"name": "A2"}]},
                {"items": [{"name": "A3"}]},
            ],
        }
    )
    result = c.compute_fill_rate()

    nested_agg = result.fields.b_list.aggregated_fields.items
    repr_str = repr(nested_agg)
    assert "FillRateNestedListAggregatedResult" in repr_str
    assert "lists=2" in repr_str


def test_aggregated_field_collection_invalid_field() -> None:
    """Test that FillRateAggregatedFieldCollection raises error for invalid field."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
            ],
        }
    )
    result = order.compute_fill_rate()

    with pytest.raises(InvalidAggregatedFieldError) as exc_info:
        _ = result.fields.items.aggregated_fields.invalid_field

    assert "invalid_field" in str(exc_info.value)


def test_aggregated_model_result_invalid_field() -> None:
    """Test that FillRateAggregatedModelResult raises error for invalid field."""

    class Address(BaseModel):
        city: str
        street: str

    class Item(BaseModel):
        name: str
        address: Address

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Item1", "address": {"city": "NYC", "street": "Main"}},
                {"name": "Item2", "address": {"city": "LA", "street": "Oak"}},
            ],
        }
    )
    result = order.compute_fill_rate()

    # Access nested model through aggregated
    address_agg = result.fields.items.aggregated_fields.address
    assert isinstance(address_agg, FillRateAggregatedModelResult)

    # Try to access invalid field
    with pytest.raises(InvalidAggregatedFieldError):
        _ = address_agg.invalid_field


def test_nested_list_aggregated_min_max() -> None:
    """Test min/max for nested list aggregation."""

    class A(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class B(BaseModel):
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                {"items": [{"score": 30.0}]},  # mean = 0.3
                {"items": [{"score": 100.0}]},  # mean = 1.0
            ],
        }
    )
    result = c.compute_fill_rate()

    nested_agg = result.fields.b_list.aggregated_fields.items

    # Hierarchical mode: means are [0.3, 1.0]
    assert nested_agg.min(hierarchical=True) == pytest.approx(0.3)
    assert nested_agg.max(hierarchical=True) == pytest.approx(1.0)

    # Flatten mode: all values are [0.3, 1.0]
    assert nested_agg.min(hierarchical=False) == pytest.approx(0.3)
    assert nested_agg.max(hierarchical=False) == pytest.approx(1.0)


def test_nested_list_aggregated_var() -> None:
    """Test var for nested list aggregation."""

    class A(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class B(BaseModel):
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                {"items": [{"score": 0.0}, {"score": 100.0}]},  # mean = 0.5
                {"items": [{"score": 100.0}]},  # mean = 1.0
            ],
        }
    )
    result = c.compute_fill_rate()

    nested_agg = result.fields.b_list.aggregated_fields.items

    # Hierarchical var: means are [0.5, 1.0], mean of means = 0.75
    # var = ((0.5-0.75)^2 + (1.0-0.75)^2) / 2 = 0.0625
    assert nested_agg.var(hierarchical=True) == pytest.approx(0.0625)


def test_nested_list_aggregated_with_single_list() -> None:
    """Test FillRateNestedListAggregatedResult with single list."""

    class A(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class B(BaseModel):
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                {"items": [{"score": 50.0}]},
            ],
        }
    )
    result = c.compute_fill_rate()

    nested_agg = result.fields.b_list.aggregated_fields.items

    # Single list
    assert nested_agg.std(hierarchical=True) == 0.0
    assert nested_agg.var(hierarchical=True) == 0.0


def test_nested_list_aggregated_quantile_invalid() -> None:
    """Test quantile with invalid q value."""

    class A(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class B(BaseModel):
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                {"items": [{"score": 50.0}]},
            ],
        }
    )
    result = c.compute_fill_rate()

    nested_agg = result.fields.b_list.aggregated_fields.items

    with pytest.raises(ValueError, match="Quantile q must be between 0.0 and 1.0"):
        _ = nested_agg.quantile(1.5, hierarchical=True)

    with pytest.raises(ValueError, match="Quantile q must be between 0.0 and 1.0"):
        _ = nested_agg.quantile(-0.1)


def test_aggregated_model_result_empty() -> None:
    """Test FillRateAggregatedModelResult with empty items."""

    from cobjectric.fill_rate import FillRateAggregatedModelResult

    empty_agg = FillRateAggregatedModelResult(_items=[])

    # Stats should return 0.0 for empty items
    assert empty_agg.mean() == 0.0
    assert empty_agg.max() == 0.0
    assert empty_agg.min() == 0.0


def test_nested_list_aggregated_empty_lists() -> None:
    """Test FillRateNestedListAggregatedResult with empty lists."""

    class A(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class B(BaseModel):
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                {"items": []},
            ],
        }
    )
    result = c.compute_fill_rate()

    nested_agg = result.fields.b_list.aggregated_fields.items

    # Empty lists
    assert nested_agg.min() == 0.0
    assert nested_agg.max() == 0.0
    assert nested_agg.min(hierarchical=True) == 0.0
    assert nested_agg.max(hierarchical=True) == 0.0


def test_nested_list_aggregated_quantile_empty_hierarchical() -> None:
    """Test quantile on empty nested list (hierarchical)."""

    class A(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class B(BaseModel):
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                {"items": []},
            ],
        }
    )
    result = c.compute_fill_rate()

    nested_agg = result.fields.b_list.aggregated_fields.items
    assert nested_agg.quantile(0.5, hierarchical=True) == 0.0


def test_aggregated_model_result_with_nested_lists() -> None:
    """Test FillRateAggregatedModelResult correctly handles nested lists."""

    class Tag(BaseModel):
        name: str

    class Item(BaseModel):
        title: str
        tags: list[Tag]

    class Catalog(BaseModel):
        items: list[Item]

    catalog = Catalog.from_dict(
        {
            "items": [
                {"title": "Item1", "tags": [{"name": "tag1"}]},
                {"title": "Item2", "tags": [{"name": "tag2"}]},
            ],
        }
    )
    result = catalog.compute_fill_rate()

    items_agg = result.fields.items.aggregated_fields
    # Access nested model
    tags_nested = items_agg.tags
    assert isinstance(tags_nested, FillRateNestedListAggregatedResult)

    # Try to access invalid field on aggregated model result
    # This should be caught by FillRateAggregatedModelResult's validation
    # (when there are nested models)


def test_nested_list_aggregated_quantile_hierarchical_multiple() -> None:
    """Test quantile hierarchical with multiple nested lists."""

    class A(BaseModel):
        score: float = Spec(fill_rate_func=lambda x: x / 100)

    class B(BaseModel):
        items: list[A]

    class C(BaseModel):
        b_list: list[B]

    c = C.from_dict(
        {
            "b_list": [
                {"items": [{"score": 0.0}, {"score": 50.0}]},  # mean = 0.25
                {"items": [{"score": 50.0}]},  # mean = 0.5
                {"items": [{"score": 100.0}]},  # mean = 1.0
            ],
        }
    )
    result = c.compute_fill_rate()

    nested_agg = result.fields.b_list.aggregated_fields.items

    # Hierarchical quantile: means are [0.25, 0.5, 1.0]
    # q=0.25: index = 0.25 * (3-1) = 0.5, interpolate between 0.25 and 0.5
    # value = 0.25 * 0.5 + 0.5 * 0.5 = 0.375
    assert nested_agg.quantile(0.25, hierarchical=True) == pytest.approx(0.375)
    # q=0.75: index = 0.75 * 2 = 1.5, interpolate between 0.5 and 1.0
    # value = 0.5 * 0.5 + 1.0 * 0.5 = 0.75
    assert nested_agg.quantile(0.75, hierarchical=True) == pytest.approx(0.75)


def test_complex_nested_example_1() -> None:
    """Test complex nested example from playground: Person with nested data."""

    class Address(BaseModel):
        street: str
        city: str

    class Experience(BaseModel):
        company: str
        title: str

    class Person(BaseModel):
        name: str
        age: int
        address: Address
        experiences: list[Experience]
        skills: list[str]

    person = Person.from_dict(
        {
            "name": "John Doe",
            "age": 30,
            "address": {"street": "123 Main St"},  # city missing
            "experiences": [
                {"company": "Google", "title": "Software Engineer"},
                {"company": "Apple", "title": "Product Manager"},
            ],
            "skills": ["Python", "JavaScript", "SQL"],
        }
    )
    result = person.compute_fill_rate()

    # Check simple fields
    assert result.fields.name.value == 1.0
    assert result.fields.age.value == 1.0

    # Check nested model with missing field
    assert result.fields.address.fields.street.value == 1.0
    assert result.fields.address.fields.city.value == 0.0

    # Check list of primitives
    assert result.fields.skills.value == 1.0

    # Check list of models
    assert isinstance(result.fields.experiences, FillRateListResult)
    assert len(result.fields.experiences) == 2

    # Check aggregated fields on list of models
    experiences_agg = result.fields.experiences.aggregated_fields
    assert isinstance(experiences_agg, FillRateAggregatedFieldCollection)
    assert experiences_agg.company.values == [1.0, 1.0]
    assert experiences_agg.title.values == [1.0, 1.0]


def test_complex_nested_example_2() -> None:
    """Test complex nested example: Order with nested Address in Item."""

    class Address(BaseModel):
        city: str
        street: str

    class Item(BaseModel):
        name: str
        address: Address

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "address": {"city": "NYC", "street": "Main"}},
                {"name": "Banana", "address": {"city": "LA"}},  # street missing
            ],
        }
    )
    result = order.compute_fill_rate()

    # Check list of models
    assert isinstance(result.fields.items, FillRateListResult)
    assert len(result.fields.items) == 2

    # Check individual items
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.address.fields.city.value == 1.0
    assert result.fields.items[0].fields.address.fields.street.value == 1.0

    assert result.fields.items[1].fields.name.value == 1.0
    assert result.fields.items[1].fields.address.fields.city.value == 1.0
    assert result.fields.items[1].fields.address.fields.street.value == 0.0

    # Check aggregated fields on list
    items_agg = result.fields.items.aggregated_fields
    assert isinstance(items_agg, FillRateAggregatedFieldCollection)
    assert items_agg.name.values == [1.0, 1.0]

    # Check aggregated nested model
    address_agg = items_agg.address
    assert isinstance(address_agg, FillRateAggregatedModelResult)
    assert address_agg.city.values == [1.0, 1.0]
    assert address_agg.street.values == [1.0, 0.0]


def test_complex_nested_example_3() -> None:
    """Test complex nested example: C with nested B containing list of A."""

    class A(BaseModel):
        name: str

    class B(BaseModel):
        items_list: list[A]

    class C(BaseModel):
        b: B
        lb: list[B]

    c = C(
        b=B(items_list=[A(name="test")]),
        lb=[B(items_list=[A(name="test2")]), B(items_list=[A(name="test3")])],
    )
    result = c.compute_fill_rate()

    # Check simple nested list access
    assert result.fields.b.fields.items_list.aggregated_fields.name.values == [1.0]

    # Check list of nested lists - single item
    assert result.fields.lb[0].fields.items_list.aggregated_fields.name.values == [1.0]

    # Check aggregated list of nested lists
    lb_agg = result.fields.lb.aggregated_fields
    assert isinstance(lb_agg, FillRateAggregatedFieldCollection)

    # Access nested list through aggregated
    items_list_nested = lb_agg.items_list
    assert isinstance(items_list_nested, FillRateNestedListAggregatedResult)
    # Two lists with 1 item each, so values are [1.0, 1.0]
    assert len(items_list_nested._lists) == 2
    assert items_list_nested.values == [1.0, 1.0]  # One value per list

    # Access field through nested list aggregation
    names_agg = items_list_nested.aggregated_fields.name
    assert names_agg.values == [1.0, 1.0]


def test_complex_nested_example_4() -> None:
    """Test complex nested with mixed missing data."""

    class Tag(BaseModel):
        name: str

    class Author(BaseModel):
        name: str
        tags: list[Tag]

    class Post(BaseModel):
        title: str
        authors: list[Author]

    class Blog(BaseModel):
        posts: list[Post]

    blog = Blog.from_dict(
        {
            "posts": [
                {
                    "title": "Post 1",
                    "authors": [
                        {
                            "name": "Alice",
                            "tags": [{"name": "python"}, {"name": "rust"}],
                        },
                        {"name": "Bob"},  # tags missing
                    ],
                },
                {
                    "title": "Post 2",
                    "authors": [
                        {"name": "Charlie", "tags": [{"name": "go"}]},
                    ],
                },
            ],
        }
    )
    result = blog.compute_fill_rate()

    # Check top level
    assert isinstance(result.fields.posts, FillRateListResult)
    assert len(result.fields.posts) == 2

    # Check post 0
    assert result.fields.posts[0].fields.title.value == 1.0
    assert len(result.fields.posts[0].fields.authors) == 2

    # Check post 0, author 0
    assert result.fields.posts[0].fields.authors[0].fields.name.value == 1.0
    assert result.fields.posts[0].fields.authors[
        0
    ].fields.tags.aggregated_fields.name.values == [1.0, 1.0]

    # Check post 0, author 1 (tags missing)
    assert result.fields.posts[0].fields.authors[1].fields.name.value == 1.0
    assert result.fields.posts[0].fields.authors[1].fields.tags.value == 0.0

    # Aggregated access
    posts_agg = result.fields.posts.aggregated_fields
    authors_nested = posts_agg.authors
    assert isinstance(authors_nested, FillRateNestedListAggregatedResult)

    # Access fields through nested list aggregation
    author_names_agg = authors_nested.aggregated_fields.name
    assert isinstance(author_names_agg, FillRateAggregatedFieldResult)
    assert author_names_agg.values == [1.0, 1.0, 1.0]  # 3 authors total

    # Nested list aggregation for tags
    tags_nested = authors_nested.aggregated_fields.tags
    assert isinstance(tags_nested, FillRateNestedListAggregatedResult)
    # Three lists: [2 tags], [] (missing), [1 tag] -> means [1.0, 0.0, 1.0]
    assert tags_nested.values == [1.0, 0.0, 1.0]


def test_coverage_fill_rate_aggregated_field_collection_nested_list() -> None:
    """Test coverage: FillRateAggregatedFieldCollection with nested lists (lines 727-731, 735-736)."""

    class A(BaseModel):
        name: str

    class B(BaseModel):
        items_list: list[A]

    class C(BaseModel):
        lb: list[B]

    c = C(
        lb=[
            B(items_list=[A(name="test1"), A(name="test2")]),
            B(items_list=[A(name="test3")]),
        ]
    )
    result = c.compute_fill_rate()

    # Access nested list through aggregated - this triggers lines 727-731, 735-736
    lb_agg = result.fields.lb.aggregated_fields
    items_list_nested = lb_agg.items_list
    assert isinstance(items_list_nested, FillRateNestedListAggregatedResult)
    assert len(items_list_nested._lists) == 2


def test_coverage_fill_rate_aggregated_field_collection_nested_model() -> None:
    """Test coverage: FillRateAggregatedFieldCollection with nested models (line 739)."""

    class Address(BaseModel):
        city: str

    class Person(BaseModel):
        address: Address

    class Company(BaseModel):
        employees: list[Person]

    company = Company(
        employees=[
            Person(address=Address(city="Paris")),
            Person(address=Address(city="London")),
        ]
    )
    result = company.compute_fill_rate()

    # Access nested model through aggregated - this triggers line 739
    employees_agg = result.fields.employees.aggregated_fields
    address_agg = employees_agg.address
    assert isinstance(address_agg, FillRateAggregatedModelResult)
    assert address_agg.city.values == [1.0, 1.0]


def test_coverage_nested_list_aggregated_result_empty_lists() -> None:
    """Test coverage: FillRateNestedListAggregatedResult with empty lists (lines 871, 956, 977, 1005)."""

    # Create an empty FillRateNestedListAggregatedResult
    empty_result = FillRateNestedListAggregatedResult([], None)

    # Test mean() with empty lists (line 871)
    assert empty_result.mean() == 0.0
    assert empty_result.mean(hierarchical=True) == 0.0

    # Test max() with empty lists (line 956)
    assert empty_result.max() == 0.0
    assert empty_result.max(hierarchical=True) == 0.0

    # Test min() with empty lists (line 977)
    assert empty_result.min() == 0.0
    assert empty_result.min(hierarchical=True) == 0.0

    # Test quantile() with empty lists (line 1005)
    assert empty_result.quantile(0.5) == 0.0
    assert empty_result.quantile(0.5, hierarchical=True) == 0.0


def test_coverage_nested_list_aggregated_result_zero_weight() -> None:
    """Test coverage: FillRateNestedListAggregatedResult with zero weight (line 884)."""

    class ZeroWeightModel(BaseModel):
        name: str = Spec(fill_rate_weight=0.0)

    class Container(BaseModel):
        items_list: list[ZeroWeightModel]

    # Create a model with items that have zero weight
    container = Container(items_list=[ZeroWeightModel(name="test")])
    result = container.compute_fill_rate()

    # Verify that the weight is actually 0.0
    assert result.fields.items_list[0].fields.name.weight == 0.0

    # Create nested list result
    nested_result = FillRateNestedListAggregatedResult(
        [result.fields.items_list], result.fields.items_list._element_type
    )

    # This should trigger line 884: we have values but total_weight == 0.0
    # The field has a value (1.0) but weight is 0.0
    # Collect values and weights to verify
    all_values, all_weights = nested_result._collect_all_values_and_weights()
    assert len(all_values) > 0  # We have values
    assert sum(all_weights) == 0.0  # But total weight is 0

    mean_val = nested_result.mean(hierarchical=False)
    assert mean_val == 0.0


def test_coverage_nested_list_aggregated_result_std_single_value() -> None:
    """Test coverage: FillRateNestedListAggregatedResult.std() with <= 1 value (line 913)."""

    class A(BaseModel):
        name: str

    class B(BaseModel):
        items_list: list[A]

    class C(BaseModel):
        lb: list[B]

    # Create a model with a single item in a single list
    # This gives us exactly 1 value when flattened
    c = C(lb=[B(items_list=[A(name="test")])])
    result = c.compute_fill_rate()

    # Access nested list aggregation
    nested_result = result.fields.lb.aggregated_fields.items_list
    assert isinstance(nested_result, FillRateNestedListAggregatedResult)

    # Test std() with single value (line 913)
    # values = [1.0] (single item), so std should return 0.0
    assert nested_result.std(hierarchical=False) == 0.0

    # Test with empty values
    empty_result = FillRateNestedListAggregatedResult([], None)
    assert empty_result.std(hierarchical=False) == 0.0


def test_coverage_nested_list_aggregated_result_var_multiple_values() -> None:
    """Test coverage: FillRateNestedListAggregatedResult.var() with multiple values."""

    class A(BaseModel):
        name: str

    class B(BaseModel):
        items_list: list[A]

    class C(BaseModel):
        lb: list[B]

    # Create a model with multiple lists
    c = C(
        lb=[
            B(items_list=[A(name="test")]),
            B(items_list=[A(name="test2")]),
        ]
    )
    result = c.compute_fill_rate()

    # Access nested list aggregation
    nested_result = result.fields.lb.aggregated_fields.items_list
    assert isinstance(nested_result, FillRateNestedListAggregatedResult)
    assert len(nested_result._lists) == 2  # We have 2 lists

    # Test var() with multiple values
    # values = [1.0, 1.0] (means of each list)
    # var = 0.0 (all values are the same)
    assert nested_result.var(hierarchical=False) == 0.0

    # Test with single list (returns 0.0 due to len(self._lists) <= 1)
    c2 = C(lb=[B(items_list=[A(name="test")])])
    result2 = c2.compute_fill_rate()
    nested_result2 = result2.fields.lb.aggregated_fields.items_list
    assert nested_result2.var(hierarchical=False) == 0.0


def test_coverage_nested_list_aggregated_result_quantile_non_hierarchical() -> None:
    """Test coverage: FillRateNestedListAggregatedResult.quantile() non-hierarchical (lines 1011, 1014)."""

    class A(BaseModel):
        name: str

    class B(BaseModel):
        items_list: list[A]

    # Create a model with multiple items
    b = B(items_list=[A(name="test1"), A(name="test2"), A(name="test3")])
    result = b.compute_fill_rate()

    nested_result = FillRateNestedListAggregatedResult(
        [result.fields.items_list], result.fields.items_list._element_type
    )

    # Test quantile() with hierarchical=False (line 1011)
    q50 = nested_result.quantile(0.5, hierarchical=False)
    assert q50 == 1.0

    # Test with empty values (line 1014)
    empty_result = FillRateNestedListAggregatedResult([], None)
    assert empty_result.quantile(0.5, hierarchical=False) == 0.0


def test_coverage_fill_rate_aggregated_field_collection_empty_items() -> None:
    """Test coverage: FillRateAggregatedFieldCollection._get_available_fields() with empty items (line 1075)."""
    from cobjectric.fill_rate import FillRateAggregatedFieldCollection

    # Create FillRateAggregatedFieldCollection with empty items
    empty_collection = FillRateAggregatedFieldCollection([])

    # This should trigger line 1075 - return [] when _items is empty
    available_fields = empty_collection._get_available_fields()
    assert available_fields == []


def test_coverage_fill_rate_list_result_collect_all_values() -> None:
    """Test coverage: FillRateListResult._collect_all_values() (lines 1228-1231)."""

    class A(BaseModel):
        name: str
        age: int

    class B(BaseModel):
        items_list: list[A]

    b = B(
        items_list=[
            A(name="test1", age=10),
            A(name="test2", age=20),
        ]
    )
    result = b.compute_fill_rate()

    # Test _collect_all_values() method (lines 1228-1231)
    all_values = result.fields.items_list._collect_all_values()
    # Should collect all values from all fields of all items
    # Each item has 2 fields (name, age), each with value 1.0
    # So we should have 4 values: [1.0, 1.0, 1.0, 1.0]
    assert len(all_values) == 4
    assert all(v == 1.0 for v in all_values)


def test_fill_rate_list_result_get_values() -> None:
    """Test that FillRateListResult._get_values() returns all values via StatsMixin."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana"},  # price missing
            ],
        }
    )
    result = order.compute_fill_rate()

    # Test _get_values() method (StatsMixin implementation)
    values = result.fields.items._get_values()
    # Should collect all values from all fields of all items
    # item 0: name=1.0, price=1.0
    # item 1: name=1.0, price=0.0
    # So we should have 4 values: [1.0, 1.0, 1.0, 0.0]
    assert len(values) == 4
    assert values == [1.0, 1.0, 1.0, 0.0]


def test_coverage_aggregated_model_result_nested_model_branch() -> None:
    """Test coverage: FillRateAggregatedModelResult.__getattr__ with nested model (lines 727-728, 739).

    This tests the case where we access a nested model through an aggregated model result.
    Structure: list[X] -> X.address (model) -> address.details (model)
    Path: result.fields.list.aggregated_fields.address.details
    """

    class Details(BaseModel):
        zip_code: str

    class Address(BaseModel):
        city: str
        details: Details

    class Person(BaseModel):
        address: Address

    class Company(BaseModel):
        employees: list[Person]

    company = Company(
        employees=[
            Person(address=Address(city="Paris", details=Details(zip_code="75000"))),
            Person(address=Address(city="London", details=Details(zip_code="SW1A"))),
        ]
    )
    result = company.compute_fill_rate()

    # Step 1: Access aggregated_fields.address -> FillRateAggregatedModelResult
    employees_agg = result.fields.employees.aggregated_fields
    address_agg = employees_agg.address
    assert isinstance(address_agg, FillRateAggregatedModelResult)

    # Step 2: Access address_agg.details -> This triggers lines 727-728, 739
    # because 'details' is a FillRateModelResult inside each Address
    details_agg = address_agg.details
    assert isinstance(details_agg, FillRateAggregatedModelResult)
    assert details_agg.zip_code.values == [1.0, 1.0]


def test_coverage_aggregated_model_result_nested_list_branch() -> None:
    """Test coverage: FillRateAggregatedModelResult.__getattr__ with nested list (lines 729-731, 735-736).

    This tests the case where we access a nested list through an aggregated model result.
    Structure: list[X] -> X.address (model) -> address.tags (list)
    Path: result.fields.list.aggregated_fields.address.tags
    """

    class Tag(BaseModel):
        name: str

    class Address(BaseModel):
        city: str
        tags: list[Tag]

    class Person(BaseModel):
        address: Address

    class Company(BaseModel):
        employees: list[Person]

    company = Company(
        employees=[
            Person(
                address=Address(
                    city="Paris",
                    tags=[Tag(name="capital"), Tag(name="europe")],
                )
            ),
            Person(
                address=Address(
                    city="London",
                    tags=[Tag(name="uk")],
                )
            ),
        ]
    )
    result = company.compute_fill_rate()

    # Step 1: Access aggregated_fields.address -> FillRateAggregatedModelResult
    employees_agg = result.fields.employees.aggregated_fields
    address_agg = employees_agg.address
    assert isinstance(address_agg, FillRateAggregatedModelResult)

    # Step 2: Access address_agg.tags -> This triggers lines 729-731, 735-736
    # because 'tags' is a FillRateListResult inside each Address
    tags_nested = address_agg.tags
    assert isinstance(tags_nested, FillRateNestedListAggregatedResult)
    # Two lists: [2 tags], [1 tag] -> means [1.0, 1.0]
    assert tags_nested.values == [1.0, 1.0]


def test_coverage_nested_list_aggregated_result_quantile_empty_values() -> None:
    """Test coverage: FillRateNestedListAggregatedResult.quantile() with empty values (lines 1011, 1014)."""

    class A(BaseModel):
        name: str

    class B(BaseModel):
        items_list: list[A]

    class C(BaseModel):
        lb: list[B]

    # Create a model with non-empty lists to have a valid nested result
    c = C(
        lb=[
            B(items_list=[A(name="test1"), A(name="test2")]),
            B(items_list=[A(name="test3")]),
        ]
    )
    result = c.compute_fill_rate()

    # Access nested list aggregation
    nested_result = result.fields.lb.aggregated_fields.items_list
    assert isinstance(nested_result, FillRateNestedListAggregatedResult)

    # Test quantile() with hierarchical=False (line 1011)
    # This calls sorted(self.values) which gives us the flattened values
    q50 = nested_result.quantile(0.5, hierarchical=False)
    assert q50 == 1.0

    # Test with empty nested list result (line 1014)
    empty_result = FillRateNestedListAggregatedResult([], None)
    q50_empty = empty_result.quantile(0.5, hierarchical=False)
    assert q50_empty == 0.0
