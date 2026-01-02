import typing as t

import pytest

from cobjectric import (
    AggregatedFieldResult,
    BaseModel,
    DuplicateFillRateAccuracyFuncError,
    InvalidFillRateValueError,
    InvalidWeightError,
    ListResult,
    MissingValue,
    ModelResult,
    Spec,
    fill_rate_accuracy_func,
)
from cobjectric.field import Field
from cobjectric.field_spec import FieldSpec


def test_default_accuracy_both_filled() -> None:
    """Test that default accuracy returns 1.0 when both got and expected are filled."""

    class Person(BaseModel):
        name: str
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person(name="Jane", age=25)

    result = person_got.compute_fill_rate_accuracy(person_expected)

    assert result.fields.name.value == 1.0
    assert result.fields.age.value == 1.0


def test_default_accuracy_both_missing() -> None:
    """Test that default accuracy returns 1.0 when both got and expected are MissingValue."""

    class Person(BaseModel):
        name: str
        age: int

    person_got = Person()
    person_expected = Person()

    result = person_got.compute_fill_rate_accuracy(person_expected)

    assert result.fields.name.value == 1.0
    assert result.fields.age.value == 1.0


def test_default_accuracy_got_filled_expected_missing() -> None:
    """Test that default accuracy returns 0.0 when got is filled but expected is missing."""

    class Person(BaseModel):
        name: str
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person()

    result = person_got.compute_fill_rate_accuracy(person_expected)

    assert result.fields.name.value == 0.0
    assert result.fields.age.value == 0.0


def test_default_accuracy_got_missing_expected_filled() -> None:
    """Test that default accuracy returns 0.0 when got is missing but expected is filled."""

    class Person(BaseModel):
        name: str
        age: int

    person_got = Person()
    person_expected = Person(name="Jane", age=25)

    result = person_got.compute_fill_rate_accuracy(person_expected)

    assert result.fields.name.value == 0.0
    assert result.fields.age.value == 0.0


def test_default_accuracy_mixed() -> None:
    """Test that default accuracy works correctly with mixed filled/missing fields."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person_got = Person(name="John", age=30)
    person_expected = Person(name="Jane", email="jane@example.com")

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # name: both filled -> 1.0
    assert result.fields.name.value == 1.0
    # age: got filled, expected missing -> 0.0
    assert result.fields.age.value == 0.0
    # email: got missing, expected filled -> 0.0
    assert result.fields.email.value == 0.0


def test_spec_fill_rate_accuracy_func() -> None:
    """Test that Spec(fill_rate_accuracy_func=...) works correctly."""

    def custom_accuracy(got: t.Any, expected: t.Any) -> float:
        if got is MissingValue and expected is MissingValue:
            return 1.0
        if got is not MissingValue and expected is not MissingValue:
            return 0.8  # Custom: both filled but not perfect
        return 0.0

    class Person(BaseModel):
        name: str = Spec(fill_rate_accuracy_func=custom_accuracy)
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person(name="Jane", age=25)

    result = person_got.compute_fill_rate_accuracy(person_expected)

    assert result.fields.name.value == 0.8
    assert result.fields.age.value == 1.0  # Default func


def test_spec_fill_rate_accuracy_weight() -> None:
    """Test that Spec(fill_rate_accuracy_weight=...) works correctly."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_accuracy_weight=2.0)
        age: int = Spec(fill_rate_accuracy_weight=0.5)

    person_got = Person(name="John", age=30)
    person_expected = Person(name="Jane", age=25)

    result = person_got.compute_fill_rate_accuracy(person_expected)

    assert result.fields.name.weight == 2.0
    assert result.fields.age.weight == 0.5


def test_decorator_accuracy_func() -> None:
    """Test that @fill_rate_accuracy_func decorator works."""

    class Person(BaseModel):
        name: str
        age: int

        @fill_rate_accuracy_func("name")
        def accuracy_name(got: t.Any, expected: t.Any) -> float:
            return (
                0.7
                if (got is not MissingValue) == (expected is not MissingValue)
                else 0.0
            )

    person_got = Person(name="John", age=30)
    person_expected = Person(name="Jane", age=25)

    result = person_got.compute_fill_rate_accuracy(person_expected)

    assert result.fields.name.value == 0.7
    assert result.fields.age.value == 1.0  # Default func


def test_decorator_accuracy_weight() -> None:
    """Test that @fill_rate_accuracy_func with weight works."""

    class Person(BaseModel):
        name: str
        age: int

        @fill_rate_accuracy_func("name", weight=2.0)
        def accuracy_name(got: t.Any, expected: t.Any) -> float:
            return (
                1.0
                if (got is not MissingValue) == (expected is not MissingValue)
                else 0.0
            )

    person_got = Person(name="John", age=30)
    person_expected = Person(name="Jane", age=25)

    result = person_got.compute_fill_rate_accuracy(person_expected)

    assert result.fields.name.weight == 2.0
    assert result.fields.age.weight == 1.0


def test_weights_are_independent() -> None:
    """Test that fill_rate_weight and fill_rate_accuracy_weight are independent."""

    class Person(BaseModel):
        name: str = Spec(
            fill_rate_func=lambda x: 0.5,
            fill_rate_weight=2.0,
            fill_rate_accuracy_weight=1.5,
        )
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person(name="Jane", age=25)

    # Fill rate result
    fill_rate_result = person_got.compute_fill_rate()
    assert fill_rate_result.fields.name.weight == 2.0

    # Accuracy result
    accuracy_result = person_got.compute_fill_rate_accuracy(person_expected)
    assert accuracy_result.fields.name.weight == 1.5


def test_duplicate_accuracy_func_raises() -> None:
    """Test that Spec + decorator on same field raises DuplicateFillRateAccuracyFuncError."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_accuracy_func=lambda got, exp: 0.5)

        @fill_rate_accuracy_func("name")
        def accuracy_name(got: t.Any, expected: t.Any) -> float:
            return 0.6

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    with pytest.raises(DuplicateFillRateAccuracyFuncError):
        person_got.compute_fill_rate_accuracy(person_expected)


def test_duplicate_accuracy_func_decorators_raises() -> None:
    """Test that multiple decorators on same field raises DuplicateFillRateAccuracyFuncError."""

    class Person(BaseModel):
        name: str

        @fill_rate_accuracy_func("name")
        def accuracy_name1(got: t.Any, expected: t.Any) -> float:
            return 0.5

        @fill_rate_accuracy_func("name")
        def accuracy_name2(got: t.Any, expected: t.Any) -> float:
            return 0.6

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    with pytest.raises(DuplicateFillRateAccuracyFuncError):
        person_got.compute_fill_rate_accuracy(person_expected)


def test_accuracy_nested_models() -> None:
    """Test that accuracy works with nested models."""

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
            "name": "Jane",
            "address": {"street": "456 Oak Ave", "city": "Somewhere"},
        }
    )

    result = person_got.compute_fill_rate_accuracy(person_expected)

    assert result.fields.name.value == 1.0
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 1.0
    assert result.fields.address.fields.city.value == 1.0


def test_accuracy_nested_missing() -> None:
    """Test that accuracy works when nested model is missing."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person(name="John")
    person_expected = Person.from_dict(
        {
            "name": "Jane",
            "address": {"street": "456 Oak Ave", "city": "Somewhere"},
        }
    )

    result = person_got.compute_fill_rate_accuracy(person_expected)

    assert result.fields.name.value == 1.0
    assert isinstance(result.fields.address, ModelResult)
    # got missing, expected filled -> 0.0
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_accuracy_mean_with_weights() -> None:
    """Test that mean() uses fill_rate_accuracy_weight for weighted mean."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_accuracy_weight=2.0)
        age: int = Spec(fill_rate_accuracy_weight=1.0)
        email: str = Spec(fill_rate_accuracy_weight=1.0)

    person_got = Person(name="John", age=30)
    person_expected = Person(name="Jane", email="jane@example.com")

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # name: both filled -> 1.0, weight=2.0
    # age: got filled, expected missing -> 0.0, weight=1.0
    # email: got missing, expected filled -> 0.0, weight=1.0
    # mean = (1.0 * 2.0 + 0.0 * 1.0 + 0.0 * 1.0) / (2.0 + 1.0 + 1.0) = 2.0 / 4.0 = 0.5
    assert result.mean() == pytest.approx(0.5)


def test_accuracy_invalid_value_raises() -> None:
    """Test that invalid accuracy value raises InvalidFillRateValueError."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_accuracy_func=lambda got, exp: 1.5)

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    with pytest.raises(InvalidFillRateValueError):
        person_got.compute_fill_rate_accuracy(person_expected)


def test_accuracy_weight_negative_in_spec_raises() -> None:
    """Test that negative fill_rate_accuracy_weight in Spec() raises InvalidWeightError."""

    with pytest.raises(InvalidWeightError) as exc_info:
        Spec(fill_rate_accuracy_weight=-1.0)

    assert "Invalid fill_rate_accuracy_weight in Spec" in str(exc_info.value)
    assert "-1.0" in str(exc_info.value)


def test_accuracy_weight_negative_in_decorator_raises() -> None:
    """Test that negative weight in decorator raises InvalidWeightError."""

    with pytest.raises(InvalidWeightError) as exc_info:

        class Person(BaseModel):
            name: str

            @fill_rate_accuracy_func("name", weight=-1.0)
            def accuracy_name(got: t.Any, expected: t.Any) -> float:
                return 1.0

    assert "Invalid fill_rate_accuracy_weight in decorator" in str(exc_info.value)
    assert "-1.0" in str(exc_info.value)


def test_accuracy_weight_zero_allowed() -> None:
    """Test that fill_rate_accuracy_weight = 0.0 is allowed."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_accuracy_weight=0.0)
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person(name="Jane", age=25)

    result = person_got.compute_fill_rate_accuracy(person_expected)

    assert result.fields.name.weight == 0.0
    assert result.fields.age.weight == 1.0


def test_accuracy_decorator_weight_overrides_spec() -> None:
    """Test that decorator weight overrides Spec fill_rate_accuracy_weight."""

    class Person(BaseModel):
        name: str = Spec(fill_rate_accuracy_weight=1.0)

        @fill_rate_accuracy_func("name", weight=2.0)
        def accuracy_name(got: t.Any, expected: t.Any) -> float:
            return (
                1.0
                if (got is not MissingValue) == (expected is not MissingValue)
                else 0.0
            )

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Decorator weight (2.0) should override Spec weight (1.0)
    assert result.fields.name.weight == 2.0


def test_accuracy_empty_model() -> None:
    """Test that empty model works correctly."""

    class EmptyModel(BaseModel):
        pass

    instance_got = EmptyModel()
    instance_expected = EmptyModel()

    result = instance_got.compute_fill_rate_accuracy(instance_expected)

    assert isinstance(result, ModelResult)
    assert result.mean() == 0.0


def test_accuracy_decorator_pattern() -> None:
    """Test that @fill_rate_accuracy_func supports glob patterns."""

    class Person(BaseModel):
        name_first: str
        name_last: str
        age: int

        @fill_rate_accuracy_func("name_*")
        def accuracy_name_fields(got: t.Any, expected: t.Any) -> float:
            return (
                0.8
                if (got is not MissingValue) == (expected is not MissingValue)
                else 0.0
            )

    person_got = Person(name_first="John", name_last="Doe", age=30)
    person_expected = Person(name_first="Jane", name_last="Smith", age=25)

    result = person_got.compute_fill_rate_accuracy(person_expected)

    assert result.fields.name_first.value == 0.8
    assert result.fields.name_last.value == 0.8
    assert result.fields.age.value == 1.0  # Default func


def test_accuracy_nested_model_type_both_missing() -> None:
    """Test accuracy when nested model type field is MissingValue in both."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Both missing -> accuracy = 1.0 for all nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 1.0
    assert result.fields.address.fields.city.value == 1.0


def test_accuracy_nested_model_type_got_missing_expected_present() -> None:
    """Test accuracy when nested model type field is MissingValue in got but present in expected."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person(name="John")
    person_expected = Person.from_dict(
        {
            "name": "Jane",
            "address": {"street": "123 Main St", "city": "Anytown"},
        }
    )

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Got missing, expected present -> accuracy = 0.0 for all nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_accuracy_nested_model_type_got_present_expected_missing() -> None:
    """Test accuracy when nested model type field is present in got but MissingValue in expected."""

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

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Got present, expected missing -> accuracy = 0.0 for all nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_accuracy_nested_model_type_expected_not_basemodel() -> None:
    """Test accuracy when expected nested model value is not a BaseModel instance."""

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
    # Manually set address to a non-BaseModel value to test the edge case
    person_expected._fields["address"] = Field(
        name="address", type=Address, value="not a model", spec=FieldSpec()
    )

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Expected is not BaseModel -> accuracy = 0.0 for all nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_accuracy_nested_both_missing() -> None:
    """Test accuracy when both got and expected have nested model missing."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person(name="John")  # address missing
    person_expected = Person(name="Jane")  # address missing

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Both missing -> accuracy = 1.0 for all nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 1.0
    assert result.fields.address.fields.city.value == 1.0


def test_accuracy_nested_one_missing() -> None:
    """Test accuracy when one has nested model and other doesn't."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person(name="John")  # address missing
    person_expected = Person.from_dict(
        {
            "name": "Jane",
            "address": {"street": "456 Oak Ave", "city": "Somewhere"},
        }
    )

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # One missing -> accuracy = 0.0 for all nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_accuracy_got_field_not_in_expected() -> None:
    """Test accuracy when got has field not in expected."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person_got = Person(name="John", age=30, email="john@example.com")
    person_expected = Person(name="Jane", age=25)

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # email in got but missing in expected -> treat expected as MissingValue
    # got.email has value, expected.email is MissingValue -> default accuracy = 0.0
    assert result.fields.email.value == 0.0


def test_accuracy_nested_expected_value_not_basemodel() -> None:
    """Test accuracy when nested field type is Address but expected is not BaseModel."""

    class Address(BaseModel):
        street: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person.from_dict(
        {"name": "John", "address": {"street": "123 Main St"}}
    )
    person_expected = Person(name="Jane")
    # Manually set address to non-BaseModel value
    person_expected._fields["address"] = Field(
        name="address", type=Address, value="not a model", spec=FieldSpec()
    )

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Expected is not BaseModel -> accuracy = 0.0 for nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0


def test_accuracy_nested_with_internal_fields() -> None:
    """Test accuracy when nested model has internal fields with underscore."""

    class Address(BaseModel):
        street: str
        _internal: str = None  # Internal field

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person(name="John")
    person_expected = Person(name="Jane")

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Both address missing -> accuracy = 1.0 for all fields
    # Internal fields (_internal) should be skipped
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 1.0


def test_accuracy_nested_both_present_got_present_expected_missing() -> None:
    """Test accuracy when nested model is present in got but missing in expected."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person.from_dict(
        {"name": "John", "address": {"street": "123 Main St", "city": "Anytown"}}
    )
    person_expected = Person(name="Jane")  # address missing

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Got has address, expected doesn't -> accuracy = 0.0 for nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_accuracy_nested_field_expected_not_basemodel() -> None:
    """Test accuracy when nested field type Field with value not being BaseModel."""

    class Address(BaseModel):
        street: str

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person.from_dict(
        {"name": "John", "address": {"street": "123 Main St"}}
    )
    person_expected = Person(name="Jane", address="not a model")  # Not a BaseModel

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Expected nested value is not BaseModel -> accuracy = 0.0
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0


def test_accuracy_field_is_basemodel_instance() -> None:
    """Test accuracy when field itself is BaseModel instance (not Field)."""

    class Address(BaseModel):
        street: str

    # Create a field that is a BaseModel instance directly
    address_instance = Address(street="123 Main St")

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    # Manually inject a BaseModel instance as a field
    person._fields["address"] = address_instance

    person_expected = Person.from_dict(
        {"name": "Jane", "address": {"street": "456 Oak Ave"}}
    )
    person_expected._fields["address"] = Address(street="456 Oak Ave")

    result = person.compute_fill_rate_accuracy(person_expected)

    # Field is BaseModel, expected is BaseModel -> compute recursively
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 1.0


def test_accuracy_field_nested_model_one_missing() -> None:
    """Test accuracy when field type is BaseModel with one missing."""

    class Address(BaseModel):
        street: str
        _internal: str = None  # Internal field to test skipping

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person(name="John")  # address missing
    person_expected = Person.from_dict({"name": "Jane", "address": {"street": "X"}})

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Got missing, expected present -> accuracy = 0.0 for nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0


def test_accuracy_field_nested_model_both_present_expected_not_basemodel() -> None:
    """Test accuracy when nested Field is present but expected is not BaseModel."""

    class Address(BaseModel):
        street: str
        _internal: str = None

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person.from_dict({"name": "John", "address": {"street": "123 Main"}})
    person_expected = Person(name="Jane")
    # Set address to string instead of BaseModel
    person_expected._fields["address"] = Field(
        name="address", type=Address, value="not a model", spec=FieldSpec()
    )

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Got is BaseModel, expected is not -> accuracy = 0.0 for nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0


def test_accuracy_field_expected_missing_in_fields() -> None:
    """Test accuracy when expected field doesn't exist in expected model."""

    class Person(BaseModel):
        name: str
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person(name="Jane")  # age missing

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Age exists in got but not in expected -> accuracy based on default func
    # Default: both not missing (got has 30, expected has MissingValue) = 0.0
    assert result.fields.age.value == 0.0


def test_accuracy_field_type_nested_both_present() -> None:
    """Test accuracy when Field type is BaseModel and both are present."""

    class Address(BaseModel):
        street: str
        _internal: str = None

    class Person(BaseModel):
        name: str
        address: Address

    person_got = Person.from_dict({"name": "John", "address": {"street": "123 Main"}})
    person_expected = Person.from_dict(
        {"name": "Jane", "address": {"street": "456 Oak"}}
    )

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Both are BaseModel -> recursively compute
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 1.0


def test_accuracy_expected_field_is_none() -> None:
    """Test accuracy when expected._fields doesn't have a field that got has."""

    class Person(BaseModel):
        name: str
        age: int

    person_got = Person(name="John", age=30)
    person_expected = Person(name="Jane", age=25)

    # Manually remove 'age' from expected._fields to simulate missing field
    del person_expected._fields["age"]

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # age exists in got but NOT in expected._fields -> expected_value = MissingValue
    # Default accuracy: got has value, expected is MissingValue -> 0.0
    assert result.fields.age.value == 0.0


def test_accuracy_nested_both_present_expected_not_basemodel_instance() -> None:
    """Test when nested Field has BaseModel value but expected is not BaseModel instance."""

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
    person_expected._fields["address"] = Field(
        name="address",
        type=Address,
        value="string_not_basemodel",
        spec=FieldSpec(),
    )

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # got.address is BaseModel, expected.address is string -> accuracy = 0.0
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_accuracy_field_is_basemodel_instance_expected_missing() -> None:
    """Test when field itself is BaseModel instance and expected is MissingValue."""

    class Address(BaseModel):
        street: str
        _internal: str = None

    class Person(BaseModel):
        name: str

    person_got = Person(name="John")
    # Inject a BaseModel instance as a field (not wrapped in Field)
    address_instance = Address(street="123 Main")
    person_got._fields["address"] = address_instance

    person_expected = Person(name="Jane")
    # expected doesn't have 'address' field at all

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # field is BaseModel, expected_field is None -> expected_value = MissingValue
    # -> accuracy = 0.0 for all nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0


def test_accuracy_field_wraps_basemodel_expected_not_basemodel() -> None:
    """Test when Field wraps a BaseModel value but expected is not BaseModel.

    This tests the case where:
    - field is a Field (not BaseModel directly)
    - field.type is a BaseModel type
    - field.value is a BaseModel instance (not MissingValue)
    - expected_value is NOT a BaseModel instance
    """

    class Address(BaseModel):
        street: str
        city: str
        _internal: str = None

    class Person(BaseModel):
        name: str
        address: Address

    # Create person_got with address as a Field containing a BaseModel
    person_got = Person(name="John")
    address_instance = Address(street="123 Main", city="NYC")
    person_got._fields["address"] = Field(
        name="address",
        type=Address,
        value=address_instance,
        spec=FieldSpec(),
    )

    # Create person_expected with address as a Field containing a non-BaseModel
    person_expected = Person(name="Jane")
    person_expected._fields["address"] = Field(
        name="address",
        type=Address,
        value="not_a_basemodel_string",
        spec=FieldSpec(),
    )

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # field.value is BaseModel, expected_value is NOT BaseModel
    # -> accuracy should be 0.0 for all nested fields
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 0.0
    assert result.fields.address.fields.city.value == 0.0


def test_accuracy_field_wraps_basemodel_both_present() -> None:
    """Test when Field wraps a BaseModel and expected is also BaseModel.

    This tests the case where both are BaseModel instances.
    """

    class Address(BaseModel):
        street: str
        _internal: str = None

    class Person(BaseModel):
        name: str
        address: Address

    # Create person_got with address as a Field containing a BaseModel
    person_got = Person(name="John")
    address_got = Address(street="123 Main")
    person_got._fields["address"] = Field(
        name="address",
        type=Address,
        value=address_got,
        spec=FieldSpec(),
    )

    # Create person_expected with address as a Field containing a BaseModel
    person_expected = Person(name="Jane")
    address_expected = Address(street="456 Oak")
    person_expected._fields["address"] = Field(
        name="address",
        type=Address,
        value=address_expected,
        spec=FieldSpec(),
    )

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Both field.value and expected_value are BaseModel
    # -> should recursively compute accuracy
    assert isinstance(result.fields.address, ModelResult)
    assert result.fields.address.fields.street.value == 1.0


def test_accuracy_list_primitive_both_filled() -> None:
    """Test accuracy for list[Primitive] when both are filled."""

    class Person(BaseModel):
        tags: list[str]

    person_got = Person(tags=["python", "rust"])
    person_expected = Person(tags=["java", "go"])

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Both have non-empty lists -> accuracy = 1.0
    assert result.fields.tags.value == 1.0


def test_accuracy_list_primitive_one_missing() -> None:
    """Test accuracy for list[Primitive] when one is missing."""

    class Person(BaseModel):
        tags: list[str]

    person_got = Person(tags=["python"])
    person_expected = Person()

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # got has list, expected is MissingValue -> accuracy = 0.0
    assert result.fields.tags.value == 0.0


def test_accuracy_list_primitive_both_empty() -> None:
    """Test accuracy for list[Primitive] when both are empty."""

    class Person(BaseModel):
        tags: list[str]

    person_got = Person(tags=[])
    person_expected = Person(tags=[])

    result = person_got.compute_fill_rate_accuracy(person_expected)

    # Both have empty lists -> accuracy = 1.0
    assert result.fields.tags.value == 1.0


def test_accuracy_list_basemodel_same_items() -> None:
    """Test accuracy for list[BaseModel] with same number of items."""

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

    # Both have 2 items, all fields filled -> accuracy = 1.0 for all
    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 2
    assert result.fields.items[0].fields.name.value == 1.0
    assert result.fields.items[0].fields.price.value == 1.0


def test_accuracy_list_basemodel_different_count() -> None:
    """Test accuracy for list[BaseModel] with different item counts."""

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

    result = order_got.compute_fill_rate_accuracy(order_expected)

    # got has 1 item, expected has 2 items
    # We compare item by item, so only first item is compared
    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 1
    assert result.fields.items[0].fields.name.value == 1.0


def test_accuracy_list_basemodel_aggregated() -> None:
    """Test that aggregated access works for accuracy results."""

    class Item(BaseModel):
        name: str
        price: float

    class Order(BaseModel):
        items: list[Item]

    order_got = Order.from_dict(
        {
            "items": [
                {"name": "Apple", "price": 1.0},
                {"name": "Banana"},  # price missing
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

    # name: both items have name -> [1.0, 1.0]
    # price: item 0 both have, item 1 got missing -> [1.0, 0.0]
    assert isinstance(result.fields.items.aggregated_fields.name, AggregatedFieldResult)
    assert result.fields.items.aggregated_fields.name.values == [1.0, 1.0]
    assert result.fields.items.aggregated_fields.price.values == [1.0, 0.0]


def test_accuracy_list_basemodel_type_mismatch() -> None:
    """Test accuracy when list item is not BaseModel instance."""

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

    result = order_got.compute_fill_rate_accuracy(order_expected)

    # Type mismatch should create empty result
    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 1
    # All fields should be 0.0 due to type mismatch
    assert result.fields.items[0].fields.name.value == 0.0


def test_accuracy_list_basemodel_type_mismatch_with_private_field() -> None:
    """Test accuracy when list item type has private fields."""

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

    result = order_got.compute_fill_rate_accuracy(order_expected)

    # Type mismatch should create empty result, private field should be skipped
    assert isinstance(result.fields.items, ListResult)
    assert len(result.fields.items) == 1
    # Only name field should be present (private field skipped)
    assert "name" in result.fields.items[0]._fields
    assert "_private" not in result.fields.items[0]._fields
