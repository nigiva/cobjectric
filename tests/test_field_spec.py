from cobjectric import BaseModel, FieldSpec, MissingValue, Spec


def test_field_spec_creation() -> None:
    """Test that FieldSpec can be created with default values."""
    spec = FieldSpec()
    assert spec.metadata == {}


def test_field_spec_with_metadata() -> None:
    """Test that FieldSpec can be created with explicit metadata."""
    metadata = {"description": "The name of the person", "required": True}
    spec = FieldSpec(metadata=metadata)
    assert spec.metadata == metadata


def test_field_spec_default_metadata_is_empty_dict() -> None:
    """Test that FieldSpec default metadata is an empty dict."""
    spec1 = FieldSpec()
    spec2 = FieldSpec()
    assert spec1.metadata == {}
    assert spec2.metadata == {}
    # Each instance should have its own dict
    spec1.metadata["key"] = "value"
    assert spec1.metadata == {"key": "value"}
    assert spec2.metadata == {}


def test_field_spec_repr() -> None:
    """Test that FieldSpec has a proper representation."""
    spec = FieldSpec(metadata={"description": "Test"})
    repr_str = repr(spec)
    assert "FieldSpec" in repr_str
    assert "description" in repr_str or "Test" in repr_str


def test_field_spec_eq() -> None:
    """Test that FieldSpec equality works correctly."""
    spec1 = FieldSpec(metadata={"description": "Test"})
    spec2 = FieldSpec(metadata={"description": "Test"})
    spec3 = FieldSpec(metadata={"description": "Different"})
    spec4 = FieldSpec()

    assert spec1 == spec2
    assert spec1 != spec3
    assert spec1 != spec4
    assert spec4 == FieldSpec()


def test_spec_returns_field_spec() -> None:
    """Test that Spec function returns a FieldSpec instance."""
    result = Spec()
    assert isinstance(result, FieldSpec)


def test_spec_with_metadata() -> None:
    """Test that Spec function accepts metadata."""
    metadata = {"description": "The name of the person"}
    result = Spec(metadata=metadata)
    assert isinstance(result, FieldSpec)
    assert result.metadata == metadata


def test_spec_without_args() -> None:
    """Test that Spec() without arguments creates FieldSpec with empty metadata."""
    result = Spec()
    assert isinstance(result, FieldSpec)
    assert result.metadata == {}


def test_spec_with_none_metadata() -> None:
    """Test that Spec(None) creates FieldSpec with empty metadata."""
    result = Spec(metadata=None)
    assert isinstance(result, FieldSpec)
    assert result.metadata == {}


def test_base_model_with_spec() -> None:
    """Test that BaseModel with Spec works correctly."""

    class Person(BaseModel):
        name: str = Spec(metadata={"description": "The name of the person"})
        age: int = Spec()
        email: str
        is_active: bool

    person = Person.from_dict(
        {
            "name": "John Doe",
            "age": 30,
            "email": "john.doe@example.com",
            "is_active": True,
        }
    )

    assert isinstance(person.fields.name.spec, FieldSpec)
    assert person.fields.name.spec.metadata == {"description": "The name of the person"}
    assert isinstance(person.fields.age.spec, FieldSpec)
    assert person.fields.age.spec.metadata == {}


def test_base_model_without_spec_has_default_field_spec() -> None:
    """Test that fields without Spec have default FieldSpec."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person.from_dict({"name": "John Doe", "age": 30})

    assert isinstance(person.fields.name.spec, FieldSpec)
    assert person.fields.name.spec.metadata == {}
    assert isinstance(person.fields.age.spec, FieldSpec)
    assert person.fields.age.spec.metadata == {}


def test_base_model_mixed_spec() -> None:
    """Test that BaseModel works with some fields having Spec and others not."""
    from cobjectric import Spec

    class Person(BaseModel):
        name: str = Spec(metadata={"description": "Name"})
        age: int
        email: str = Spec(metadata={"format": "email"})
        is_active: bool

    person = Person.from_dict(
        {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "is_active": True,
        }
    )

    assert person.fields.name.spec.metadata == {"description": "Name"}
    assert person.fields.age.spec.metadata == {}
    assert person.fields.email.spec.metadata == {"format": "email"}
    assert person.fields.is_active.spec.metadata == {}


def test_nested_model_with_spec() -> None:
    """Test that nested models work with Spec."""
    from cobjectric import Spec

    class Address(BaseModel):
        street: str = Spec(metadata={"description": "Street address"})
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

    assert isinstance(person.fields.address, Address)
    assert person.fields.address.fields.street.spec.metadata == {
        "description": "Street address"
    }
    assert person.fields.address.fields.city.spec.metadata == {}


def test_list_field_with_spec() -> None:
    """Test that list fields work with Spec."""
    from cobjectric import Spec

    class Person(BaseModel):
        name: str
        skills: list[str] = Spec(metadata={"description": "List of skills"})

    person = Person.from_dict({"name": "John Doe", "skills": ["Python", "JavaScript"]})

    assert person.fields.skills.spec.metadata == {"description": "List of skills"}
    assert person.fields.skills.value == ["Python", "JavaScript"]


def test_dict_field_with_spec() -> None:
    """Test that dict fields work with Spec."""
    from cobjectric import Spec

    class Person(BaseModel):
        name: str
        metadata: dict = Spec(metadata={"description": "Additional metadata"})

    person = Person.from_dict({"name": "John Doe", "metadata": {"key": "value"}})

    assert person.fields.metadata.spec.metadata == {
        "description": "Additional metadata"
    }
    assert person.fields.metadata.value == {"key": "value"}


def test_typed_dict_field_with_spec() -> None:
    """Test that typed dict fields work with Spec."""
    from cobjectric import Spec

    class Person(BaseModel):
        name: str
        scores: dict[str, int] = Spec(metadata={"description": "Test scores"})

    person = Person.from_dict(
        {"name": "John Doe", "scores": {"math": 90, "english": 85}}
    )

    assert person.fields.scores.spec.metadata == {"description": "Test scores"}
    assert person.fields.scores.value == {"math": 90, "english": 85}


def test_union_field_with_spec() -> None:
    """Test that union fields work with Spec."""
    from cobjectric import Spec

    class Person(BaseModel):
        name: str
        id: str | int = Spec(metadata={"description": "User ID"})

    person1 = Person.from_dict({"name": "John Doe", "id": "abc123"})
    person2 = Person.from_dict({"name": "Jane Doe", "id": 123})

    assert person1.fields.id.spec.metadata == {"description": "User ID"}
    assert person1.fields.id.value == "abc123"
    assert person2.fields.id.spec.metadata == {"description": "User ID"}
    assert person2.fields.id.value == 123


def test_spec_does_not_affect_value() -> None:
    """Test that Spec does not affect the field value."""
    from cobjectric import Spec

    class Person(BaseModel):
        name: str = Spec(metadata={"description": "Name"})
        age: int = Spec()

    person = Person.from_dict({"name": "John Doe", "age": 30})

    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value == 30
    assert person.fields.name.spec.metadata == {"description": "Name"}
    assert person.fields.age.spec.metadata == {}


def test_spec_with_missing_field() -> None:
    """Test that Spec works with missing fields."""
    from cobjectric import Spec

    class Person(BaseModel):
        name: str = Spec(metadata={"description": "Name"})
        age: int

    person = Person.from_dict({"name": "John Doe"})

    assert person.fields.name.value == "John Doe"
    assert person.fields.name.spec.metadata == {"description": "Name"}
    assert person.fields.age.value is MissingValue
    assert isinstance(person.fields.age.spec, FieldSpec)
    assert person.fields.age.spec.metadata == {}


def test_spec_with_invalid_type() -> None:
    """Test that Spec works even when field value is invalid."""
    from cobjectric import Spec

    class Person(BaseModel):
        name: str = Spec(metadata={"description": "Name"})
        age: int = Spec()

    person = Person.from_dict({"name": "John Doe", "age": "invalid"})

    assert person.fields.name.value == "John Doe"
    assert person.fields.name.spec.metadata == {"description": "Name"}
    assert person.fields.age.value is MissingValue
    assert isinstance(person.fields.age.spec, FieldSpec)
    assert person.fields.age.spec.metadata == {}
