import typing as t

import pytest

from cobjectric import BaseModel, FieldContext, FieldSpec, MissingValue, Spec


def test_field_context_creation() -> None:
    """Test that FieldContext can be created."""
    spec = FieldSpec()
    context = FieldContext(name="age", field_type=int, spec=spec)
    assert context.name == "age"
    assert context.field_type is int
    assert context.spec is spec


def test_field_context_repr() -> None:
    """Test that FieldContext has a proper representation."""
    spec = FieldSpec()
    context = FieldContext(name="age", field_type=int, spec=spec)
    repr_str = repr(context)
    assert "FieldContext" in repr_str
    assert "age" in repr_str


def test_simple_normalizer_works() -> None:
    """Test that simple normalizer (1 param) still works."""

    class Person(BaseModel):
        name: str = Spec(normalizer=lambda x: x.lower())

    person = Person(name="JOHN")
    assert person.fields.name.value == "john"


def test_contextual_normalizer_receives_context() -> None:
    """Test that contextual normalizer (2 params) receives FieldContext."""

    def contextual_normalizer(value: t.Any, context: FieldContext) -> t.Any:
        assert context.name == "age"
        assert context.field_type is int
        assert isinstance(context.spec, FieldSpec)
        if context.field_type is int:
            return int(float(value))
        return value

    class Person(BaseModel):
        age: int = Spec(normalizer=contextual_normalizer)

    person = Person(age=30.0)
    assert person.fields.age.value == 30
    assert isinstance(person.fields.age.value, int)


def test_contextual_normalizer_with_union_type() -> None:
    """Test contextual normalizer with Union type (int | None)."""

    def contextual_normalizer(value: t.Any, context: FieldContext) -> t.Any:
        if value is None:
            return None
        # Extract base type from Union - check if int is in the union
        import types

        origin = t.get_origin(context.field_type)
        if origin is t.Union or origin is types.UnionType:
            args = t.get_args(context.field_type)
            if int in args:
                return int(float(value))
        elif context.field_type is int:
            return int(float(value))
        return value

    class Person(BaseModel):
        age: int | None = Spec(normalizer=contextual_normalizer)

    person1 = Person(age=30.0)
    assert person1.fields.age.value == 30
    assert isinstance(person1.fields.age.value, int)

    person2 = Person(age=None)
    assert person2.fields.age.value is None


def test_contextual_normalizer_with_decorator() -> None:
    """Test that contextual normalizer works with @field_normalizer decorator."""

    from cobjectric import field_normalizer

    class Person(BaseModel):
        age: int

        @field_normalizer("age")
        def normalize_age(value: t.Any, context: FieldContext) -> t.Any:
            assert context.name == "age"
            assert context.field_type is int
            if context.field_type is int:
                return int(float(value))
            return value

    person = Person(age=30.0)
    assert person.fields.age.value == 30
    assert isinstance(person.fields.age.value, int)


def test_contextual_normalizer_combined_with_spec() -> None:
    """Test contextual normalizer combined with Spec normalizer."""

    def spec_normalizer(value: t.Any, context: FieldContext) -> t.Any:
        if context.field_type is int:
            return int(float(value))
        return value

    from cobjectric import field_normalizer

    class Person(BaseModel):
        age: int = Spec(normalizer=spec_normalizer)

        @field_normalizer("age")
        def double_normalizer(value: t.Any, context: FieldContext) -> t.Any:
            # This runs after spec_normalizer
            assert isinstance(value, int)
            return value * 2

    person = Person(age=15.0)
    # spec_normalizer: 15.0 -> 15 (int)
    # double_normalizer: 15 -> 30
    assert person.fields.age.value == 30


def test_contextual_normalizer_missing_value() -> None:
    """Test that contextual normalizer is not called for MissingValue."""

    def contextual_normalizer(value: t.Any, context: FieldContext) -> t.Any:
        assert False, "Should not be called for MissingValue"

    class Person(BaseModel):
        age: int = Spec(normalizer=contextual_normalizer)

    person = Person()
    assert person.fields.age.value is MissingValue


def test_contextual_normalizer_with_nested_model() -> None:
    """Test contextual normalizer with nested BaseModel."""

    def contextual_normalizer(value: t.Any, context: FieldContext) -> t.Any:
        assert context.name == "age"
        assert context.field_type is int
        if context.field_type is int:
            return int(float(value))
        return value

    class Address(BaseModel):
        street: str

    class Person(BaseModel):
        age: int = Spec(normalizer=contextual_normalizer)
        address: Address

    person = Person(age=30.0, address={"street": "Main St"})
    assert person.fields.age.value == 30
    assert isinstance(person.fields.age.value, int)
    assert person.fields.address.fields.street.value == "Main St"


def test_contextual_normalizer_with_list() -> None:
    """Test contextual normalizer with list field."""

    def contextual_normalizer(value: t.Any, context: FieldContext) -> t.Any:
        assert context.name == "ages"
        # Check if it's a list type
        origin = t.get_origin(context.field_type)
        assert origin is list
        if isinstance(value, list):
            return [int(float(v)) for v in value]
        return value

    class Person(BaseModel):
        ages: list[int] = Spec(normalizer=contextual_normalizer)

    person = Person(ages=[30.0, 25.0, 40.0])
    assert person.fields.ages.value == [30, 25, 40]
    assert all(isinstance(v, int) for v in person.fields.ages.value)


def test_build_combined_normalizer_contextual_without_context() -> None:
    """Test that _build_combined_normalizer raises error if contextual normalizer without context."""

    def contextual_normalizer(value: t.Any, context: FieldContext) -> t.Any:
        return value

    # This should never happen in normal usage, but we test the error case
    from cobjectric.base_model import BaseModel

    with pytest.raises(
        ValueError,
        match="Contextual normalizer requires FieldContext but context is None",
    ):
        BaseModel._build_combined_normalizer(
            spec_normalizer=contextual_normalizer,
            decorator_normalizers=[],
            context=None,
        )
