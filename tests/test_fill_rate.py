import typing as t

import pytest

from cobjectric import (
    BaseModel,
    DuplicateFillRateFuncError,
    FillRateFieldResult,
    FillRateModelResult,
    InvalidFillRateValueError,
    InvalidWeightError,
    MissingValue,
    Spec,
    fill_rate_func,
)


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


def test_fill_rate_field_collection_parse_path_with_brackets() -> None:
    """Test parsing paths with list index brackets."""

    class Item(BaseModel):
        name: str

    class Order(BaseModel):
        items: list[Item]

    order = Order.from_dict({"items": [{"name": "Item1"}, {"name": "Item2"}]})
    result = order.compute_fill_rate()

    # Test _parse_path with brackets
    segments = result.fields._parse_path("items[0].name")
    assert segments == ["items", "[0]", "name"]


def test_fill_rate_field_collection_parse_path_simple() -> None:
    """Test parsing simple paths."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()

    segments = result.fields._parse_path("name")
    assert segments == ["name"]


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


def test_fill_rate_field_collection_parse_invalid_bracket() -> None:
    """Test parsing path with unclosed bracket raises KeyError."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()

    # Unclosed bracket
    with pytest.raises(KeyError, match="Invalid path"):
        _ = result.fields._parse_path("name[0")


def test_fill_rate_field_collection_parse_non_numeric_bracket() -> None:
    """Test parsing path with non-numeric bracket raises KeyError."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    result = person.compute_fill_rate()

    # Non-numeric index
    with pytest.raises(KeyError, match="Invalid path"):
        _ = result.fields._parse_path("name[abc]")
