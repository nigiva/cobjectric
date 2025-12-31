import typing as t

import pytest

from cobjectric import BaseModel, MissingValue, Spec, field_normalizer


def test_spec_normalizer_lowercase() -> None:
    """Test that Spec normalizer converts value to lowercase."""

    class Person(BaseModel):
        name: str = Spec(normalizer=lambda x: x.lower())

    person = Person(name="JOHN DOE")
    assert person.fields.name.value == "john doe"


def test_spec_normalizer_type_conversion() -> None:
    """Test that Spec normalizer can convert types."""

    class Person(BaseModel):
        age: int = Spec(normalizer=lambda x: int(x))

    person = Person(age="30")
    assert person.fields.age.value == 30
    assert isinstance(person.fields.age.value, int)


def test_spec_normalizer_incompatible_type() -> None:
    """Test that normalizer returning incompatible type results in MissingValue."""

    class Person(BaseModel):
        age: int = Spec(normalizer=lambda x: "not an int")

    person = Person(age=30)
    assert person.fields.age.value is MissingValue


def test_spec_normalizer_exception_propagates() -> None:
    """Test that normalizer exceptions propagate."""

    class Person(BaseModel):
        age: int = Spec(normalizer=lambda x: int(x) / 0)

    with pytest.raises(ZeroDivisionError):
        Person(age="30")


def test_spec_normalizer_multiple_fields() -> None:
    """Test that multiple fields can have different normalizers."""

    class Person(BaseModel):
        name: str = Spec(normalizer=lambda x: x.lower())
        age: int = Spec(normalizer=lambda x: int(x))

    person = Person(name="JOHN DOE", age="30")
    assert person.fields.name.value == "john doe"
    assert person.fields.age.value == 30


def test_decorator_normalizer_single_field() -> None:
    """Test that @field_normalizer works with a single field."""

    class Person(BaseModel):
        name: str

        @field_normalizer("name")
        def normalize_name(x: t.Any) -> str:
            return str(x).strip().lower()

    person = Person(name="  JOHN DOE  ")
    assert person.fields.name.value == "john doe"


def test_decorator_normalizer_multiple_fields() -> None:
    """Test that @field_normalizer works with multiple fields."""

    class Person(BaseModel):
        first_name: str
        last_name: str

        @field_normalizer("first_name", "last_name")
        def normalize_names(x: t.Any) -> str:
            return str(x).strip().title()

    person = Person(first_name="  john  ", last_name="  DOE  ")
    assert person.fields.first_name.value == "John"
    assert person.fields.last_name.value == "Doe"


def test_decorator_normalizer_multiple_decorators() -> None:
    """Test that multiple decorators on same field are applied in order."""

    class Person(BaseModel):
        name: str

        @field_normalizer("name")
        def trim(x: t.Any) -> str:
            return str(x).strip()

        @field_normalizer("name")
        def lowercase(x: t.Any) -> str:
            return str(x).lower()

    person = Person(name="  JOHN DOE  ")
    # First trim, then lowercase
    assert person.fields.name.value == "john doe"


def test_decorator_normalizer_pattern_matching() -> None:
    """Test that @field_normalizer supports glob patterns."""

    class Person(BaseModel):
        name_first: str
        name_last: str
        age: int

        @field_normalizer("name_*")
        def normalize_name_fields(x: t.Any) -> str:
            return str(x).strip().title()

    person = Person(name_first="  john  ", name_last="  DOE  ", age=30)
    assert person.fields.name_first.value == "John"
    assert person.fields.name_last.value == "Doe"
    assert person.fields.age.value == 30


def test_combined_spec_and_decorator() -> None:
    """Test that Spec normalizer runs first, then decorator normalizers."""

    class Person(BaseModel):
        name: str = Spec(normalizer=lambda x: x.lower())

        @field_normalizer("name")
        def trim(x: t.Any) -> str:
            return str(x).strip()

    person = Person(name="  JOHN DOE  ")
    # First Spec normalizer (lowercase), then decorator (trim)
    assert person.fields.name.value == "john doe"


def test_combined_multiple_normalizers() -> None:
    """Test chain of 3+ normalizers."""

    class Person(BaseModel):
        name: str = Spec(normalizer=lambda x: x.lower())

        @field_normalizer("name")
        def trim(x: t.Any) -> str:
            return str(x).strip()

        @field_normalizer("name")
        def capitalize_first(x: t.Any) -> str:
            return str(x).capitalize()

    person = Person(name="  JOHN DOE  ")
    # lowercase -> trim -> capitalize_first
    assert person.fields.name.value == "John doe"


def test_spec_normalizer_stored_in_field() -> None:
    """Test that spec.normalizer contains the combined normalizer."""

    def my_normalizer(x: t.Any) -> str:
        return str(x).lower()

    class Person(BaseModel):
        name: str = Spec(normalizer=my_normalizer)

    person = Person(name="JOHN")
    assert person.fields.name.spec.normalizer is not None
    assert person.fields.name.spec.normalizer("TEST") == "test"


def test_combined_normalizer_stored_in_field() -> None:
    """Test that combined normalizer is stored in spec.normalizer."""

    class Person(BaseModel):
        name: str = Spec(normalizer=lambda x: x.lower())

        @field_normalizer("name")
        def trim(x: t.Any) -> str:
            return str(x).strip()

    person = Person(name="  JOHN  ")
    assert person.fields.name.spec.normalizer is not None
    # Combined normalizer should apply both transformations
    result = person.fields.name.spec.normalizer("  TEST  ")
    assert result == "test"


def test_normalizer_with_missing_value() -> None:
    """Test that normalizer is not applied when value is MissingValue."""

    class Person(BaseModel):
        name: str = Spec(normalizer=lambda x: x.lower())

    person = Person()
    assert person.fields.name.value is MissingValue


def test_normalizer_on_optional_field() -> None:
    """Test normalizer on optional field str | None."""

    class Person(BaseModel):
        email: str | None = Spec(normalizer=lambda x: x.lower() if x else None)

    person1 = Person(email="JOHN@EXAMPLE.COM")
    assert person1.fields.email.value == "john@example.com"

    person2 = Person(email=None)
    assert person2.fields.email.value is None


def test_normalizer_on_list_field() -> None:
    """Test normalizer on list field."""

    class Person(BaseModel):
        tags: list[str] = Spec(normalizer=lambda x: [t.lower() for t in x])

    person = Person(tags=["TAG1", "TAG2", "TAG3"])
    assert person.fields.tags.value == ["tag1", "tag2", "tag3"]


def test_normalizer_on_dict_field() -> None:
    """Test normalizer on dict field."""

    class Person(BaseModel):
        metadata: dict = Spec(normalizer=lambda x: {k.lower(): v for k, v in x.items()})

    person = Person(metadata={"KEY1": "value1", "KEY2": "value2"})
    assert person.fields.metadata.value == {"key1": "value1", "key2": "value2"}


def test_normalizer_from_dict() -> None:
    """Test that normalizers work with from_dict."""

    class Person(BaseModel):
        name: str = Spec(normalizer=lambda x: x.lower())

    person = Person.from_dict({"name": "JOHN DOE"})
    assert person.fields.name.value == "john doe"


def test_decorator_normalizer_from_dict() -> None:
    """Test that decorator normalizers work with from_dict."""

    class Person(BaseModel):
        name: str

        @field_normalizer("name")
        def normalize(x: t.Any) -> str:
            return str(x).strip().lower()

    person = Person.from_dict({"name": "  JOHN DOE  "})
    assert person.fields.name.value == "john doe"


def test_normalizer_with_error_handling() -> None:
    """Test that user can handle errors in their normalizer."""

    def safe_int_normalizer(x: t.Any) -> int | None:
        try:
            return int(x)
        except (ValueError, TypeError):
            return None  # Will become MissingValue due to type mismatch with int

    class Person(BaseModel):
        age: int = Spec(normalizer=safe_int_normalizer)

    person = Person(age="invalid")
    assert person.fields.age.value is MissingValue


def test_normalizer_empty_class() -> None:
    """Test that empty class without normalizers works."""

    class Person(BaseModel):
        name: str

    person = Person(name="John")
    assert person.fields.name.value == "John"
    assert person.fields.name.spec.normalizer is None


def test_normalizer_nested_model() -> None:
    """Test normalizer on nested BaseModel field."""

    class Address(BaseModel):
        street: str = Spec(normalizer=lambda x: x.strip().title())

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {
            "name": "John",
            "address": {"street": "  123 main st  "},
        }
    )
    assert person.fields.address.fields.street.value == "123 Main St"


def test_normalizer_pattern_no_match() -> None:
    """Test that pattern matching doesn't match unrelated fields."""

    class Person(BaseModel):
        name_first: str
        name_last: str
        age: int

        @field_normalizer("name_*")
        def normalize(x: t.Any) -> str:
            return str(x).upper()

    person = Person(name_first="john", name_last="doe", age=30)
    assert person.fields.name_first.value == "JOHN"
    assert person.fields.name_last.value == "DOE"
    assert person.fields.age.value == 30


def test_normalizer_multiple_patterns() -> None:
    """Test that a normalizer can match multiple patterns."""

    class Person(BaseModel):
        first_name: str
        last_name: str
        middle_name: str

        @field_normalizer("first_*", "last_*")
        def normalize(x: t.Any) -> str:
            return str(x).title()

    person = Person(first_name="john", last_name="doe", middle_name="middle")
    assert person.fields.first_name.value == "John"
    assert person.fields.last_name.value == "Doe"
    assert person.fields.middle_name.value == "middle"


def test_normalizer_returns_missing_value() -> None:
    """Test that normalizer can explicitly return MissingValue."""

    class Person(BaseModel):
        name: str = Spec(normalizer=lambda x: MissingValue if not x else x)

    person1 = Person(name="John")
    assert person1.fields.name.value == "John"

    person2 = Person(name="")
    assert person2.fields.name.value is MissingValue


def test_normalizer_with_typed_dict() -> None:
    """Test normalizer with typed dict field."""

    class Person(BaseModel):
        scores: dict[str, int] = Spec(
            normalizer=lambda x: {k.lower(): v for k, v in x.items()}
        )

    person = Person(scores={"MATH": 90, "ENGLISH": 85})
    assert person.fields.scores.value == {"math": 90, "english": 85}


def test_normalizer_order_preserved() -> None:
    """Test that normalizer order is preserved across multiple decorators."""

    class Person(BaseModel):
        name: str

        @field_normalizer("name")
        def add_prefix(x: t.Any) -> str:
            return f"PREFIX_{x}"

        @field_normalizer("name")
        def add_suffix(x: t.Any) -> str:
            return f"{x}_SUFFIX"

    person = Person(name="test")
    # Order: add_prefix -> add_suffix
    assert person.fields.name.value == "PREFIX_test_SUFFIX"


def test_normalizer_with_union_type() -> None:
    """Test normalizer with union type field."""

    class Person(BaseModel):
        id: str | int = Spec(
            normalizer=lambda x: str(x).upper() if isinstance(x, str) else x
        )

    person1 = Person(id="abc123")
    assert person1.fields.id.value == "ABC123"

    person2 = Person(id=123)
    assert person2.fields.id.value == 123


def test_normalizer_returns_missing_value_keeps_missing() -> None:
    """Test that when normalizer returns MissingValue, field stays MissingValue."""

    class Person(BaseModel):
        name: str = Spec(normalizer=lambda x: MissingValue)

    person = Person(name="John")
    assert person.fields.name.value is MissingValue


def test_decorator_normalizer_returns_missing_value() -> None:
    """Test that decorator normalizer returning MissingValue keeps MissingValue."""

    class Person(BaseModel):
        name: str

        @field_normalizer("name")
        def normalize(x: t.Any) -> t.Any:
            return MissingValue

    person = Person(name="John")
    assert person.fields.name.value is MissingValue


def test_normalizer_type_mismatch_after_normalization() -> None:
    """Test that type mismatch after normalization results in MissingValue."""

    class Person(BaseModel):
        age: int = Spec(normalizer=lambda x: "not an int")

    person = Person(age=30)
    assert person.fields.age.value is MissingValue


def test_decorator_normalizer_type_mismatch() -> None:
    """Test that decorator normalizer causing type mismatch results in MissingValue."""

    class Person(BaseModel):
        age: int

        @field_normalizer("age")
        def normalize(x: t.Any) -> str:
            return "not an int"

    person = Person(age=30)
    assert person.fields.age.value is MissingValue


def test_combined_normalizer_type_mismatch() -> None:
    """Test that combined normalizers causing type mismatch results in MissingValue."""

    class Person(BaseModel):
        age: int = Spec(normalizer=lambda x: str(x))

        @field_normalizer("age")
        def normalize(x: t.Any) -> str:
            return f"age_{x}"

    person = Person(age=30)
    # After normalization: "age_30" (str), but field type is int -> MissingValue
    assert person.fields.age.value is MissingValue


def test_normalizer_returns_missing_value_in_chain() -> None:
    """Test that MissingValue returned in normalizer chain keeps MissingValue."""

    class Person(BaseModel):
        name: str = Spec(normalizer=lambda x: x.lower())

        @field_normalizer("name")
        def normalize(x: t.Any) -> t.Any:
            if x == "invalid":
                return MissingValue
            return x

    person1 = Person(name="John")
    assert person1.fields.name.value == "john"

    person2 = Person(name="INVALID")
    # First normalizer converts to lowercase, second returns MissingValue
    assert person2.fields.name.value is MissingValue
