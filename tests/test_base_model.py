import typing as t
from unittest.mock import MagicMock, patch

import pytest

from cobjectric import BaseModel, MissingValue


def test_base_model_creation_with_kwargs() -> None:
    """Test that BaseModel can be created with kwargs."""

    class Person(BaseModel):
        name: str
        age: int
        email: str
        is_active: bool

    person = Person(
        name="John Doe",
        age=30,
        email="john.doe@example.com",
        is_active=True,
    )
    assert person is not None


def test_base_model_fields_access() -> None:
    """Test that BaseModel fields can be accessed via .fields."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John Doe", age=30)
    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value == 30


def test_base_model_fields_are_readonly() -> None:
    """Test that BaseModel fields cannot be modified after creation."""

    class Person(BaseModel):
        name: str

    person = Person(name="John Doe")
    with pytest.raises(AttributeError):
        person.name = "Jane Doe"


def test_base_model_missing_field_has_missing_value() -> None:
    """Test that missing fields have MissingValue."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John Doe")
    assert person.fields.age.value is MissingValue


def test_base_model_invalid_type_has_missing_value() -> None:
    """Test that fields with invalid type have MissingValue."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="John Doe", age="invalid")
    assert person.fields.age.value is MissingValue


def test_base_model_valid_types() -> None:
    """Test that BaseModel accepts valid types."""

    class Person(BaseModel):
        name: str
        age: int
        is_active: bool

    person = Person(
        name="John Doe",
        age=30,
        is_active=True,
    )
    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value == 30
    assert person.fields.is_active.value is True


def test_base_model_multiple_instances() -> None:
    """Test that multiple BaseModel instances work independently."""

    class Person(BaseModel):
        name: str
        age: int

    person1 = Person(name="John", age=30)
    person2 = Person(name="Jane", age=25)
    assert person1.fields.name.value == "John"
    assert person2.fields.name.value == "Jane"


def test_base_model_ignores_private_fields() -> None:
    """Test that BaseModel ignores fields starting with underscore."""

    class Person(BaseModel):
        name: str
        _private_field: str

    person = Person(name="John")
    assert person.fields.name.value == "John"
    with pytest.raises(AttributeError):
        _ = person.fields._private_field


def test_base_model_without_annotations() -> None:
    """Test that BaseModel can be instantiated without field annotations."""

    class EmptyModel(BaseModel):
        pass

    instance = EmptyModel()
    assert instance is not None
    assert len(list(instance.fields)) == 0


def test_base_model_without_annotations_with_kwargs() -> None:
    """Test that BaseModel without annotations ignores kwargs."""

    class EmptyModel(BaseModel):
        pass

    instance = EmptyModel(extra_field="value")
    assert instance is not None
    assert len(list(instance.fields)) == 0


def test_from_dict_basic() -> None:
    """Test that from_dict creates an instance from a dictionary."""

    class Person(BaseModel):
        name: str
        age: int
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
    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value == 30
    assert person.fields.email.value == "john.doe@example.com"
    assert person.fields.is_active.value is True


def test_from_dict_missing_field() -> None:
    """Test that missing fields in dict result in MissingValue."""

    class Person(BaseModel):
        name: str
        age: int
        email: str

    person = Person.from_dict({"name": "John Doe", "age": 30})
    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value == 30
    assert person.fields.email.value is MissingValue


def test_from_dict_invalid_type() -> None:
    """Test that invalid types in dict result in MissingValue."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person.from_dict({"name": "John Doe", "age": "invalid"})
    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value is MissingValue


def test_from_dict_empty_dict() -> None:
    """Test that from_dict works with an empty dictionary."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person.from_dict({})
    assert person.fields.name.value is MissingValue
    assert person.fields.age.value is MissingValue


def test_from_dict_extra_keys() -> None:
    """Test that extra keys in dict are ignored."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person.from_dict({"name": "John Doe", "age": 30, "extra_field": "ignored"})
    assert person.fields.name.value == "John Doe"
    assert person.fields.age.value == 30
    with pytest.raises(AttributeError):
        _ = person.fields.extra_field


def test_nested_model_from_dict() -> None:
    """Test that nested models can be created from dict."""

    class Address(BaseModel):
        street: str
        city: str
        state: str
        zip_code: str
        country: str

    class Person(BaseModel):
        name: str
        age: int
        email: str
        is_active: bool
        address: Address

    person = Person.from_dict(
        {
            "name": "John Doe",
            "age": 30,
            "email": "john.doe@example.com",
            "is_active": True,
            "address": {
                "street": "123 Main St",
                "city": "Anytown",
                "state": "CA",
                "zip_code": "12345",
                "country": "USA",
            },
        }
    )
    assert isinstance(person.fields.address, Address)
    assert person.fields.address.fields.street.value == "123 Main St"
    assert person.fields.address.fields.city.value == "Anytown"
    assert person.fields.address.fields.state.value == "CA"
    assert person.fields.address.fields.zip_code.value == "12345"
    assert person.fields.address.fields.country.value == "USA"


def test_nested_model_access_via_fields() -> None:
    """Test that nested model fields can be accessed via .fields."""

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
    assert person.fields.name.value == "John Doe"
    assert isinstance(person.fields.address, Address)
    assert person.fields.address.fields.street.value == "123 Main St"
    assert person.fields.address.fields.city.value == "Anytown"


def test_nested_model_with_kwargs() -> None:
    """Test that nested models can be created with instance in kwargs."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    address = Address(street="123 Main St", city="Anytown")
    person = Person(name="John Doe", address=address)
    assert person.fields.name.value == "John Doe"
    assert isinstance(person.fields.address, Address)
    assert person.fields.address.fields.street.value == "123 Main St"
    assert person.fields.address.fields.city.value == "Anytown"


def test_nested_model_missing_value() -> None:
    """Test that missing nested model results in MissingValue."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict({"name": "John Doe"})
    assert person.fields.name.value == "John Doe"
    assert person.fields.address.value is MissingValue


def test_nested_model_invalid_type() -> None:
    """Test that invalid type for nested model results in MissingValue."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict({"name": "John Doe", "address": "invalid"})
    assert person.fields.name.value == "John Doe"
    assert person.fields.address.value is MissingValue


def test_deeply_nested_models() -> None:
    """Test that deeply nested models work correctly."""

    class Country(BaseModel):
        name: str
        code: str

    class Address(BaseModel):
        street: str
        city: str
        country: Country

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {
            "name": "John Doe",
            "address": {
                "street": "123 Main St",
                "city": "Anytown",
                "country": {"name": "United States", "code": "US"},
            },
        }
    )
    assert person.fields.name.value == "John Doe"
    assert isinstance(person.fields.address, Address)
    assert person.fields.address.fields.street.value == "123 Main St"
    assert isinstance(person.fields.address.fields.country, Country)
    assert person.fields.address.fields.country.fields.name.value == "United States"
    assert person.fields.address.fields.country.fields.code.value == "US"


def test_nested_model_with_invalid_dict() -> None:
    """Test nested model with dict that has invalid field types."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {
            "name": "John Doe",
            "address": {"street": "123 Main St", "city": 123},
        }
    )
    assert person.fields.name.value == "John Doe"
    assert isinstance(person.fields.address, Address)
    assert person.fields.address.fields.street.value == "123 Main St"
    assert person.fields.address.fields.city.value is MissingValue


def test_list_str_field() -> None:
    """Test that list[str] fields work correctly."""

    class Person(BaseModel):
        name: str
        skills: list[str]

    person = Person(name="John Doe", skills=["Python", "JavaScript", "Rust"])
    assert person.fields.name.value == "John Doe"
    assert person.fields.skills.value == ["Python", "JavaScript", "Rust"]


def test_list_int_field() -> None:
    """Test that list[int] fields work correctly."""

    class Person(BaseModel):
        name: str
        scores: list[int]

    person = Person(name="John Doe", scores=[85, 90, 95])
    assert person.fields.name.value == "John Doe"
    assert person.fields.scores.value == [85, 90, 95]


def test_list_float_field() -> None:
    """Test that list[float] fields work correctly."""

    class Person(BaseModel):
        name: str
        prices: list[float]

    person = Person(name="John Doe", prices=[10.5, 20.3, 30.7])
    assert person.fields.prices.value == [10.5, 20.3, 30.7]


def test_list_bool_field() -> None:
    """Test that list[bool] fields work correctly."""

    class Person(BaseModel):
        name: str
        flags: list[bool]

    person = Person(name="John Doe", flags=[True, False, True])
    assert person.fields.flags.value == [True, False, True]


def test_list_empty() -> None:
    """Test that empty lists work correctly."""

    class Person(BaseModel):
        name: str
        skills: list[str]

    person = Person(name="John Doe", skills=[])
    assert person.fields.skills.value == []


def test_list_missing() -> None:
    """Test that missing list fields have MissingValue."""

    class Person(BaseModel):
        name: str
        skills: list[str]

    person = Person(name="John Doe")
    assert person.fields.skills.value is MissingValue


def test_list_invalid_type() -> None:
    """Test that list filters out invalid element types."""

    class Person(BaseModel):
        name: str
        scores: list[int]

    person = Person(name="John Doe", scores=[1, 2, "invalid"])
    assert person.fields.name.value == "John Doe"
    assert person.fields.scores.value == [1, 2]


def test_list_from_dict() -> None:
    """Test that list fields work with from_dict."""

    class Person(BaseModel):
        name: str
        skills: list[str]

    person = Person.from_dict({"name": "John Doe", "skills": ["Python", "JavaScript"]})
    assert person.fields.name.value == "John Doe"
    assert person.fields.skills.value == ["Python", "JavaScript"]


def test_list_model_field() -> None:
    """Test that list[BaseModel] fields work correctly."""

    class Experience(BaseModel):
        title: str
        company: str
        start_date: str
        end_date: str
        description: str

    class Person(BaseModel):
        name: str
        experiences: list[Experience]

    person = Person.from_dict(
        {
            "name": "John Doe",
            "experiences": [
                {
                    "title": "Software Engineer",
                    "company": "Tech Corp",
                    "start_date": "2020-01-01",
                    "end_date": "2022-01-01",
                    "description": "Worked on Python projects",
                },
                {
                    "title": "Senior Engineer",
                    "company": "Big Tech",
                    "start_date": "2022-01-01",
                    "end_date": "2024-01-01",
                    "description": "Led a team",
                },
            ],
        }
    )
    assert person.fields.name.value == "John Doe"
    assert len(person.fields.experiences.value) == 2

    exp0 = person.fields.experiences.value[0]
    assert isinstance(exp0, Experience)
    assert exp0.fields.title.value == "Software Engineer"
    assert exp0.fields.company.value == "Tech Corp"
    assert exp0.fields.start_date.value == "2020-01-01"
    assert exp0.fields.end_date.value == "2022-01-01"
    assert exp0.fields.description.value == "Worked on Python projects"

    exp1 = person.fields.experiences.value[1]
    assert isinstance(exp1, Experience)
    assert exp1.fields.title.value == "Senior Engineer"
    assert exp1.fields.company.value == "Big Tech"
    assert exp1.fields.start_date.value == "2022-01-01"
    assert exp1.fields.end_date.value == "2024-01-01"
    assert exp1.fields.description.value == "Led a team"


def test_list_model_with_instances() -> None:
    """Test that list[BaseModel] works with instances."""

    class Experience(BaseModel):
        title: str
        company: str

    class Person(BaseModel):
        name: str
        experiences: list[Experience]

    exp1 = Experience(title="Engineer", company="Tech Corp")
    exp2 = Experience(title="Senior", company="Big Tech")
    person = Person(name="John Doe", experiences=[exp1, exp2])
    assert person.fields.name.value == "John Doe"
    assert len(person.fields.experiences.value) == 2
    assert person.fields.experiences.value[0] is exp1
    assert person.fields.experiences.value[0].fields.title.value == "Engineer"
    assert person.fields.experiences.value[0].fields.company.value == "Tech Corp"
    assert person.fields.experiences.value[1] is exp2
    assert person.fields.experiences.value[1].fields.title.value == "Senior"
    assert person.fields.experiences.value[1].fields.company.value == "Big Tech"


def test_list_model_with_invalid_dict() -> None:
    """Test that list[BaseModel] filters out invalid dict elements."""

    class Experience(BaseModel):
        title: str
        company: str

    class Person(BaseModel):
        name: str
        experiences: list[Experience]

    person = Person.from_dict(
        {
            "name": "John Doe",
            "experiences": [
                {"title": "Engineer", "company": "Tech Corp"},
                {"title": 123, "company": "Big Tech"},
            ],
        }
    )
    assert person.fields.name.value == "John Doe"
    assert len(person.fields.experiences.value) == 2
    assert isinstance(person.fields.experiences.value[0], Experience)
    assert person.fields.experiences.value[0].fields.title.value == "Engineer"
    assert person.fields.experiences.value[0].fields.company.value == "Tech Corp"
    assert isinstance(person.fields.experiences.value[1], Experience)
    assert person.fields.experiences.value[1].fields.title.value is MissingValue
    assert person.fields.experiences.value[1].fields.company.value == "Big Tech"


def test_list_model_missing() -> None:
    """Test that missing list[BaseModel] has MissingValue."""

    class Experience(BaseModel):
        title: str
        company: str

    class Person(BaseModel):
        name: str
        experiences: list[Experience]

    person = Person.from_dict({"name": "John Doe"})
    assert person.fields.experiences.value is MissingValue


def test_list_union_type_raises_error() -> None:
    """Test that list with Union type raises UnsupportedListTypeError."""

    from cobjectric import UnsupportedListTypeError

    class Person(BaseModel):
        name: str
        mixed: list[str | int]

    with pytest.raises(UnsupportedListTypeError) as exc_info:
        Person(name="John Doe", mixed=["a", 1])

    assert "Unsupported list type" in str(exc_info.value)
    assert "list[str | int]" in str(exc_info.value) or "list[typing.Union" in str(
        exc_info.value
    )
    assert "single type" in str(exc_info.value)


def test_list_field_not_a_list() -> None:
    """Test that non-list value for list field has MissingValue."""

    class Person(BaseModel):
        name: str
        skills: list[str]

    person = Person(name="John Doe", skills="not a list")
    assert person.fields.skills.value is MissingValue


def test_list_model_invalid_item_type() -> None:
    """Test that list[BaseModel] filters out non-dict items."""

    class Experience(BaseModel):
        title: str
        company: str

    class Person(BaseModel):
        name: str
        experiences: list[Experience]

    person = Person.from_dict(
        {
            "name": "John Doe",
            "experiences": [
                {"title": "Engineer", "company": "Tech Corp"},
                "invalid string",
            ],
        }
    )
    assert person.fields.name.value == "John Doe"
    assert len(person.fields.experiences.value) == 1
    assert isinstance(person.fields.experiences.value[0], Experience)
    assert person.fields.experiences.value[0].fields.title.value == "Engineer"
    assert person.fields.experiences.value[0].fields.company.value == "Tech Corp"


def test_list_model_from_dict_exception() -> None:
    """Test that list[BaseModel] filters out items that raise exceptions."""

    class Experience(BaseModel):
        title: str
        company: str

    class Person(BaseModel):
        name: str
        experiences: list[Experience]

    with patch.object(
        Experience, "from_dict", side_effect=ValueError("Test exception")
    ):
        person = Person.from_dict(
            {
                "name": "John Doe",
                "experiences": [
                    {"title": "Engineer", "company": "Tech Corp"},
                    {"title": "Engineer", "company": "Tech Corp"},
                ],
            }
        )
        assert person.fields.name.value == "John Doe"
        assert person.fields.experiences.value is MissingValue


def test_list_without_type_args_raises_error() -> None:
    """Test that list without type arguments raises MissingListTypeArgError."""

    from cobjectric import MissingListTypeArgError

    class Person(BaseModel):
        name: str
        items: list  # type: ignore[type-arg]

    with pytest.raises(MissingListTypeArgError) as exc_info:
        Person(name="John Doe", items=["a", 1, True])

    assert "List type must specify an element type" in str(exc_info.value)


def test_list_without_type_args_t_list_raises_error() -> None:
    """Test that t.List without type arguments raises MissingListTypeArgError."""

    from cobjectric import MissingListTypeArgError

    class Person(BaseModel):
        name: str
        items: t.List  # type: ignore[type-arg]

    with pytest.raises(MissingListTypeArgError) as exc_info:
        Person(name="John Doe", items=["a", 1, True])

    assert "List type must specify an element type" in str(exc_info.value)


def test_list_partial_filtering_keeps_valid() -> None:
    """Test that list[int] filters out invalid elements and keeps valid ones."""

    class Person(BaseModel):
        name: str
        scores: list[int]

    person = Person(name="John Doe", scores=[1, "a", 2, "b", 3, 4.5, 5])
    assert person.fields.name.value == "John Doe"
    assert person.fields.scores.value == [1, 2, 3, 5]


def test_list_partial_filtering_all_invalid() -> None:
    """Test that list[int] with all invalid elements becomes MissingValue."""

    class Person(BaseModel):
        name: str
        scores: list[int]

    person = Person(name="John Doe", scores=["a", "b", "c"])
    assert person.fields.name.value == "John Doe"
    assert person.fields.scores.value is MissingValue


def test_list_str_all_invalid() -> None:
    """Test that list[str] with all invalid elements becomes MissingValue."""

    class Person(BaseModel):
        name: str
        skills: list[str]

    person = Person(name="John Doe", skills=[1, 2, 3])
    assert person.fields.name.value == "John Doe"
    assert person.fields.skills.value is MissingValue


def test_list_model_all_non_dict() -> None:
    """Test that list[Model] with all non-dict elements becomes MissingValue."""

    class Experience(BaseModel):
        title: str
        company: str

    class Person(BaseModel):
        name: str
        experiences: list[Experience]

    person = Person.from_dict(
        {
            "name": "John Doe",
            "experiences": ["invalid1", "invalid2", 123],
        }
    )
    assert person.fields.name.value == "John Doe"
    assert person.fields.experiences.value is MissingValue


def test_list_model_mixed_dict_string() -> None:
    """Test that list[Model] keeps dicts and filters out strings."""

    class Experience(BaseModel):
        title: str
        company: str

    class Person(BaseModel):
        name: str
        experiences: list[Experience]

    person = Person.from_dict(
        {
            "name": "John Doe",
            "experiences": [
                {"title": "Engineer", "company": "Tech Corp"},
                "invalid string",
                {"title": "Senior", "company": "Big Tech"},
                "another invalid",
            ],
        }
    )
    assert person.fields.name.value == "John Doe"
    assert len(person.fields.experiences.value) == 2
    assert person.fields.experiences.value[0].fields.title.value == "Engineer"
    assert person.fields.experiences.value[0].fields.company.value == "Tech Corp"
    assert person.fields.experiences.value[1].fields.title.value == "Senior"
    assert person.fields.experiences.value[1].fields.company.value == "Big Tech"


def test_unsupported_type_any() -> None:
    """Test that t.Any as field type raises UnsupportedTypeError."""

    from cobjectric import UnsupportedTypeError

    class Person(BaseModel):
        name: str
        data: t.Any

    with pytest.raises(UnsupportedTypeError) as exc_info:
        Person(name="John Doe", data="test")

    assert "Unsupported type" in str(exc_info.value)
    assert "Any" in str(exc_info.value)


def test_unsupported_type_object() -> None:
    """Test that object as field type raises UnsupportedTypeError."""

    from cobjectric import UnsupportedTypeError

    class Person(BaseModel):
        name: str
        data: object

    with pytest.raises(UnsupportedTypeError) as exc_info:
        Person(name="John Doe", data="test")

    assert "Unsupported type" in str(exc_info.value)
    assert "object" in str(exc_info.value)


def test_unsupported_type_set() -> None:
    """Test that set[str] raises UnsupportedTypeError."""

    from cobjectric import UnsupportedTypeError

    class Person(BaseModel):
        name: str
        tags: set[str]

    with pytest.raises(UnsupportedTypeError) as exc_info:
        Person(name="John Doe", tags={"tag1", "tag2"})

    assert "Unsupported type" in str(exc_info.value)


def test_unsupported_type_tuple() -> None:
    """Test that tuple[str, int] raises UnsupportedTypeError."""

    from cobjectric import UnsupportedTypeError

    class Person(BaseModel):
        name: str
        pair: tuple[str, int]

    with pytest.raises(UnsupportedTypeError) as exc_info:
        Person(name="John Doe", pair=("a", 1))

    assert "Unsupported type" in str(exc_info.value)


def test_dict_type_supported() -> None:
    """Test that dict is supported as a primitive type."""

    class Person(BaseModel):
        name: str
        data: dict

    person = Person(name="John Doe", data={"key": "value", "age": 30})
    assert person.fields.name.value == "John Doe"
    assert person.fields.data.value == {"key": "value", "age": 30}
    assert isinstance(person.fields.data.value, dict)


def test_dict_type_from_dict() -> None:
    """Test that dict type works with from_dict."""

    class Person(BaseModel):
        name: str
        metadata: dict

    person = Person.from_dict(
        {
            "name": "John Doe",
            "metadata": {"key1": "value1", "key2": 42},
        }
    )
    assert person.fields.name.value == "John Doe"
    assert person.fields.metadata.value == {"key1": "value1", "key2": 42}


def test_dict_type_invalid_value() -> None:
    """Test that non-dict value for dict field has MissingValue."""

    class Person(BaseModel):
        name: str
        data: dict

    person = Person(name="John Doe", data="not a dict")
    assert person.fields.name.value == "John Doe"
    assert person.fields.data.value is MissingValue


def test_list_model_with_extra_kwargs() -> None:
    """Test that extra keys in dict are ignored for models with lists."""

    class Experience(BaseModel):
        title: str
        company: str

    class Person(BaseModel):
        name: str
        experiences: list[Experience]

    person = Person.from_dict(
        {
            "name": "John Doe",
            "age": 30,
            "unknown_field": "ignored",
            "experiences": [
                {
                    "title": "Engineer",
                    "company": "Tech Corp",
                    "salary": 100000,
                },
            ],
        }
    )

    assert person.fields.name.value == "John Doe"
    assert len(person.fields.experiences.value) == 1
    assert person.fields.experiences.value[0].fields.title.value == "Engineer"
    assert person.fields.experiences.value[0].fields.company.value == "Tech Corp"

    with pytest.raises(AttributeError):
        _ = person.fields.age
    with pytest.raises(AttributeError):
        _ = person.fields.experiences.value[0].fields.salary


def test_is_list_type_with_bare_list() -> None:
    """Test that _is_list_type returns True for bare list type."""

    assert BaseModel._is_list_type(list) is True


def test_get_list_element_type_without_args() -> None:
    """Test that _get_list_element_type raises error for list without args."""

    from cobjectric import MissingListTypeArgError

    with pytest.raises(MissingListTypeArgError):
        BaseModel._get_list_element_type(list)


def test_validate_field_type_list_origin_no_args() -> None:
    """Test that _validate_field_type raises error for list origin without args."""

    from cobjectric import MissingListTypeArgError

    class Person(BaseModel):
        name: str
        items: list  # type: ignore[type-arg]

    with pytest.raises(MissingListTypeArgError):
        Person(name="John", items=[])


def test_validate_field_type_list_origin_no_args_direct() -> None:
    """Test _validate_field_type directly with a mock type that has list origin but no args."""

    from cobjectric import MissingListTypeArgError

    mock_type = MagicMock()
    mock_type.__class__ = type
    with patch("typing.get_origin", return_value=list):
        with patch("typing.get_args", return_value=()):
            with pytest.raises(MissingListTypeArgError):
                BaseModel._validate_field_type(mock_type)


def test_process_nested_model_value_with_dict_and_base_model() -> None:
    """Test _process_nested_model_value with dict value and BaseModel field_type."""

    class Address(BaseModel):
        street: str
        city: str

    result = BaseModel._process_nested_model_value(
        {"street": "123 Main St", "city": "Paris"}, Address
    )
    assert isinstance(result, Address)
    assert result.fields.street.value == "123 Main St"
    assert result.fields.city.value == "Paris"


def test_process_nested_model_value_with_instance() -> None:
    """Test _process_nested_model_value with BaseModel instance."""

    class Address(BaseModel):
        street: str
        city: str

    address = Address(street="123 Main St", city="Paris")
    result = BaseModel._process_nested_model_value(address, Address)
    assert result is address


def test_process_nested_model_value_with_invalid_value() -> None:
    """Test _process_nested_model_value with invalid value."""

    class Address(BaseModel):
        street: str
        city: str

    result = BaseModel._process_nested_model_value("invalid", Address)
    assert result is MissingValue


def test_process_nested_model_value_with_dict_and_primitive_type() -> None:
    """Test _process_nested_model_value with dict value but primitive field_type."""

    result = BaseModel._process_nested_model_value({"key": "value"}, int)
    assert result is MissingValue


def test_process_nested_model_value_with_dict_type_and_dict_value() -> None:
    """Test _process_nested_model_value with dict type and dict value returns the value."""

    result = BaseModel._process_nested_model_value({"key": "value"}, dict)
    assert result == {"key": "value"}


def test_process_nested_model_value_with_dict_type_and_non_dict_value() -> None:
    """Test _process_nested_model_value with dict type but non-dict value."""

    result = BaseModel._process_nested_model_value("not a dict", dict)
    assert result is MissingValue


def test_nested_base_model_with_instance() -> None:
    """Test BaseModel with nested BaseModel instance."""

    class User(BaseModel):
        name: str
        age: int

    class Company(BaseModel):
        name: str
        owner: User

    user = User(name="John", age=30)
    company = Company(name="Tech Corp", owner=user)
    assert company.fields.name.value == "Tech Corp"
    assert isinstance(company.fields.owner, User)
    assert company.fields.owner.fields.name.value == "John"
    assert company.fields.owner.fields.age.value == 30


def test_nested_base_model_with_dict() -> None:
    """Test BaseModel with nested BaseModel from dict."""

    class User(BaseModel):
        name: str
        age: int

    class Company(BaseModel):
        name: str
        owner: User

    company = Company(name="Tech Corp", owner={"name": "John", "age": 30})
    assert company.fields.name.value == "Tech Corp"
    assert isinstance(company.fields.owner, User)
    assert company.fields.owner.fields.name.value == "John"
    assert company.fields.owner.fields.age.value == 30


def test_nested_base_model_from_dict() -> None:
    """Test BaseModel.from_dict with nested BaseModel."""

    class User(BaseModel):
        name: str
        age: int

    class Company(BaseModel):
        name: str
        owner: User

    company = Company.from_dict(
        {
            "name": "Tech Corp",
            "owner": {"name": "John", "age": 30},
        }
    )
    assert company.fields.name.value == "Tech Corp"
    assert isinstance(company.fields.owner, User)
    assert company.fields.owner.fields.name.value == "John"
    assert company.fields.owner.fields.age.value == 30


def test_dict_field_with_dict_value() -> None:
    """Test BaseModel with dict field and dict value."""

    class Config(BaseModel):
        name: str
        metadata: dict

    config = Config(name="test", metadata={"key": "value", "count": 42})
    assert config.fields.name.value == "test"
    assert config.fields.metadata.value == {"key": "value", "count": 42}


def test_dict_field_with_base_model_instance() -> None:
    """Test BaseModel with dict field but BaseModel instance value."""

    class User(BaseModel):
        name: str

    class Config(BaseModel):
        name: str
        metadata: dict

    user = User(name="John")
    config = Config(name="test", metadata=user)
    assert config.fields.name.value == "test"
    assert config.fields.metadata.value is MissingValue


def test_primitive_field_with_base_model_instance() -> None:
    """Test BaseModel with primitive field but BaseModel instance value."""

    class User(BaseModel):
        name: str

    class Person(BaseModel):
        name: str
        age: int

    user = User(name="John")
    person = Person(name="Jane", age=user)
    assert person.fields.name.value == "Jane"
    assert person.fields.age.value is MissingValue


def test_primitive_field_with_dict_value() -> None:
    """Test BaseModel with primitive field but dict value."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person(name="Jane", age={"key": "value"})
    assert person.fields.name.value == "Jane"
    assert person.fields.age.value is MissingValue


def test_primitive_field_with_dict_from_dict() -> None:
    """Test BaseModel.from_dict with primitive field but dict value."""

    class Person(BaseModel):
        name: str
        age: int

    person = Person.from_dict({"name": "Jane", "age": {"key": "value"}})
    assert person.fields.name.value == "Jane"
    assert person.fields.age.value is MissingValue


def test_deeply_nested_base_model_with_mixed_values() -> None:
    """Test deeply nested BaseModel with mixed values."""

    class Country(BaseModel):
        name: str
        code: str

    class Address(BaseModel):
        street: str
        city: str
        country: Country

    class Person(BaseModel):
        name: str
        address: Address

    person = Person.from_dict(
        {
            "name": "John",
            "address": {
                "street": "123 Main St",
                "city": "Paris",
                "country": {"name": "France", "code": "FR"},
            },
        }
    )
    assert person.fields.name.value == "John"
    assert isinstance(person.fields.address, Address)
    assert person.fields.address.fields.street.value == "123 Main St"
    assert person.fields.address.fields.city.value == "Paris"
    assert isinstance(person.fields.address.fields.country, Country)
    assert person.fields.address.fields.country.fields.name.value == "France"
    assert person.fields.address.fields.country.fields.code.value == "FR"


def test_nested_base_model_with_invalid_dict() -> None:
    """Test nested BaseModel with invalid dict (not matching BaseModel structure)."""

    class User(BaseModel):
        name: str
        age: int

    class Company(BaseModel):
        name: str
        owner: User

    company = Company(name="Tech Corp", owner={"invalid": "data"})
    assert company.fields.name.value == "Tech Corp"
    assert isinstance(company.fields.owner, User)
    assert company.fields.owner.fields.name.value is MissingValue
    assert company.fields.owner.fields.age.value is MissingValue


def test_dict_field_with_nested_dict() -> None:
    """Test dict field with nested dict structure."""

    class Config(BaseModel):
        name: str
        settings: dict

    config = Config(
        name="app",
        settings={
            "database": {"host": "localhost", "port": 5432},
            "cache": {"enabled": True},
        },
    )
    assert config.fields.name.value == "app"
    assert config.fields.settings.value == {
        "database": {"host": "localhost", "port": 5432},
        "cache": {"enabled": True},
    }


def test_string_field_with_base_model_instance() -> None:
    """Test string field with BaseModel instance."""

    class User(BaseModel):
        name: str

    class Person(BaseModel):
        name: str
        description: str

    user = User(name="John")
    person = Person(name="Jane", description=user)
    assert person.fields.name.value == "Jane"
    assert person.fields.description.value is MissingValue


def test_string_field_with_dict_value() -> None:
    """Test string field with dict value."""

    class Person(BaseModel):
        name: str
        description: str

    person = Person(name="Jane", description={"key": "value"})
    assert person.fields.name.value == "Jane"
    assert person.fields.description.value is MissingValue


def test_bool_field_with_base_model_instance() -> None:
    """Test bool field with BaseModel instance."""

    class User(BaseModel):
        name: str

    class Person(BaseModel):
        name: str
        is_active: bool

    user = User(name="John")
    person = Person(name="Jane", is_active=user)
    assert person.fields.name.value == "Jane"
    assert person.fields.is_active.value is MissingValue


def test_bool_field_with_dict_value() -> None:
    """Test bool field with dict value."""

    class Person(BaseModel):
        name: str
        is_active: bool

    person = Person(name="Jane", is_active={"key": "value"})
    assert person.fields.name.value == "Jane"
    assert person.fields.is_active.value is MissingValue


def test_float_field_with_base_model_instance() -> None:
    """Test float field with BaseModel instance."""

    class User(BaseModel):
        name: str

    class Product(BaseModel):
        name: str
        price: float

    user = User(name="John")
    product = Product(name="Widget", price=user)
    assert product.fields.name.value == "Widget"
    assert product.fields.price.value is MissingValue


def test_float_field_with_dict_value() -> None:
    """Test float field with dict value."""

    class Product(BaseModel):
        name: str
        price: float

    product = Product(name="Widget", price={"key": "value"})
    assert product.fields.name.value == "Widget"
    assert product.fields.price.value is MissingValue


def test_nested_base_model_with_string_value() -> None:
    """Test nested BaseModel field with string value."""

    class User(BaseModel):
        name: str
        age: int

    class Company(BaseModel):
        name: str
        owner: User

    company = Company(name="Tech Corp", owner="invalid")
    assert company.fields.name.value == "Tech Corp"
    assert company.fields.owner.value is MissingValue


def test_nested_base_model_with_int_value() -> None:
    """Test nested BaseModel field with int value."""

    class User(BaseModel):
        name: str
        age: int

    class Company(BaseModel):
        name: str
        owner: User

    company = Company(name="Tech Corp", owner=42)
    assert company.fields.name.value == "Tech Corp"
    assert company.fields.owner.value is MissingValue


def test_complex_nested_with_dict_field() -> None:
    """Test complex nested structure with dict field."""

    class Metadata(BaseModel):
        version: str
        config: dict

    class App(BaseModel):
        name: str
        metadata: Metadata

    app = App.from_dict(
        {
            "name": "MyApp",
            "metadata": {
                "version": "1.0.0",
                "config": {"key1": "value1", "key2": 42},
            },
        }
    )
    assert app.fields.name.value == "MyApp"
    assert isinstance(app.fields.metadata, Metadata)
    assert app.fields.metadata.fields.version.value == "1.0.0"
    assert app.fields.metadata.fields.config.value == {"key1": "value1", "key2": 42}
