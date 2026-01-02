import pytest

from cobjectric import BaseModel, MissingValue


def test_keyword_spec_import() -> None:
    """Test that KeywordSpec can be imported."""
    from cobjectric.specs import KeywordSpec

    assert KeywordSpec is not None


def test_keyword_spec_strip_default() -> None:
    """Test KeywordSpec with default strip=True."""

    from cobjectric.specs import KeywordSpec

    class Person(BaseModel):
        id: str = KeywordSpec()

    person = Person(id="  ABC123  ")
    assert person.fields.id.value == "ABC123"


def test_keyword_spec_strip_false() -> None:
    """Test KeywordSpec with strip=False."""

    from cobjectric.specs import KeywordSpec

    class Person(BaseModel):
        id: str = KeywordSpec(strip=False)

    person = Person(id="  ABC123  ")
    assert person.fields.id.value == "  ABC123  "


def test_keyword_spec_similarity_exact() -> None:
    """Test that KeywordSpec uses exact similarity."""

    from cobjectric.specs import KeywordSpec

    class Person(BaseModel):
        id: str = KeywordSpec()

    person_got = Person(id="ABC123")
    person_expected = Person(id="ABC123")

    result = person_got.compute_similarity(person_expected)
    assert result.fields.id.value == 1.0

    person_expected2 = Person(id="ABC124")
    result2 = person_got.compute_similarity(person_expected2)
    assert result2.fields.id.value == 0.0


def test_keyword_spec_convert_int_to_str_default() -> None:
    """Test KeywordSpec with default convert_int_to_str=True."""

    from cobjectric.specs import KeywordSpec

    class Person(BaseModel):
        id: str = KeywordSpec()

    person = Person(id=123)
    assert person.fields.id.value == "123"
    assert isinstance(person.fields.id.value, str)


def test_keyword_spec_convert_int_to_str_false() -> None:
    """Test KeywordSpec with convert_int_to_str=False."""

    from cobjectric.specs import KeywordSpec

    class Person(BaseModel):
        id: str = KeywordSpec(convert_int_to_str=False)

    # Without conversion, int stays as int and fails type validation
    person = Person(id=123)
    assert person.fields.id.value is MissingValue


def test_keyword_spec_convert_int_and_strip() -> None:
    """Test KeywordSpec converts int to string and strips whitespace."""

    from cobjectric.specs import KeywordSpec

    class Person(BaseModel):
        id: str = KeywordSpec()

    person1 = Person(id=123)
    assert person1.fields.id.value == "123"

    person2 = Person(id="  ABC123  ")
    assert person2.fields.id.value == "ABC123"


def test_keyword_spec_convert_int_no_strip() -> None:
    """Test KeywordSpec converts int but doesn't strip if strip=False."""

    from cobjectric.specs import KeywordSpec

    class Person(BaseModel):
        id: str = KeywordSpec(strip=False)

    person = Person(id=123)
    assert person.fields.id.value == "123"


def test_keyword_spec_custom_parameters() -> None:
    """Test KeywordSpec with custom Spec parameters."""

    from cobjectric.specs import KeywordSpec

    class Person(BaseModel):
        id: str = KeywordSpec(
            strip=True,
            similarity_weight=2.0,
            metadata={"description": "User ID"},
        )

    person = Person(id="  ABC123  ")
    assert person.fields.id.value == "ABC123"
    assert person.fields.id.spec.metadata == {"description": "User ID"}

    result = person.compute_similarity(Person(id="ABC123"))
    assert result.fields.id.weight == 2.0


def test_text_spec_import() -> None:
    """Test that TextSpec can be imported."""
    from cobjectric.specs import TextSpec

    assert TextSpec is not None


def test_text_spec_default_normalization() -> None:
    """Test TextSpec with default normalization (lower, strip, etc.)."""

    from cobjectric.specs import TextSpec

    class Person(BaseModel):
        name: str = TextSpec()

    person = Person(name="  JOHN DOE  ")
    # Should be lowercased and stripped
    assert person.fields.name.value == "john doe"


def test_text_spec_lower_false() -> None:
    """Test TextSpec with lower=False."""

    from cobjectric.specs import TextSpec

    class Person(BaseModel):
        name: str = TextSpec(lower=False)

    person = Person(name="  JOHN DOE  ")
    # Should be stripped but not lowercased
    assert person.fields.name.value == "JOHN DOE"


def test_text_spec_strip_false() -> None:
    """Test TextSpec with strip=False."""

    from cobjectric.specs import TextSpec

    class Person(BaseModel):
        name: str = TextSpec(strip=False)

    person = Person(name="  JOHN DOE  ")
    # Should be lowercased but not stripped
    assert person.fields.name.value == "  john doe  "


def test_text_spec_fuzzy_similarity() -> None:
    """Test that TextSpec uses fuzzy similarity."""

    from cobjectric.specs import TextSpec

    class Person(BaseModel):
        name: str = TextSpec()

    person_got = Person(name="John Doe")
    person_expected = Person(name="john doe")

    result = person_got.compute_similarity(person_expected)
    # Fuzzy match should give high similarity despite case difference
    assert result.fields.name.value > 0.9


def test_text_spec_custom_scorer() -> None:
    """Test TextSpec with custom scorer."""

    from cobjectric.specs import TextSpec

    class Person(BaseModel):
        name: str = TextSpec(scorer="ratio")

    person = Person(name="John Doe")
    assert person.fields.name.value == "john doe"


def test_numeric_spec_import() -> None:
    """Test that NumericSpec can be imported."""
    from cobjectric.specs import NumericSpec

    assert NumericSpec is not None


def test_numeric_spec_int_coercion() -> None:
    """Test NumericSpec with int field (coerces 30.0 -> 30)."""

    from cobjectric.specs import NumericSpec

    class Person(BaseModel):
        age: int = NumericSpec()

    person = Person(age=30.0)
    assert person.fields.age.value == 30
    assert isinstance(person.fields.age.value, int)


def test_numeric_spec_float_coercion() -> None:
    """Test NumericSpec with float field (coerces 30 -> 30.0)."""

    from cobjectric.specs import NumericSpec

    class Person(BaseModel):
        score: float = NumericSpec()

    person = Person(score=85)
    assert person.fields.score.value == 85.0
    assert isinstance(person.fields.score.value, float)


def test_numeric_spec_coerce_type_false() -> None:
    """Test NumericSpec with coerce_type=False."""

    from cobjectric.specs import NumericSpec

    class Person(BaseModel):
        age: int = NumericSpec(coerce_type=False)

    person = Person(age=30.0)
    # Without coercion, 30.0 stays as float and fails type validation
    assert person.fields.age.value is MissingValue


def test_numeric_spec_max_difference() -> None:
    """Test NumericSpec with max_difference for similarity."""

    from cobjectric.specs import NumericSpec

    class Person(BaseModel):
        score: float = NumericSpec(max_difference=5.0)

    person_got = Person(score=10.0)
    person_expected = Person(score=12.0)

    result = person_got.compute_similarity(person_expected)
    # diff=2, 2/5=0.4, 1-0.4=0.6
    assert result.fields.score.value == pytest.approx(0.6)


def test_numeric_spec_union_type() -> None:
    """Test NumericSpec with Union type (int | None)."""

    from cobjectric.specs import NumericSpec

    class Person(BaseModel):
        age: int | None = NumericSpec()

    person1 = Person(age=30.0)
    assert person1.fields.age.value == 30
    assert isinstance(person1.fields.age.value, int)

    person2 = Person(age=None)
    assert person2.fields.age.value is None


def test_boolean_spec_import() -> None:
    """Test that BooleanSpec can be imported."""
    from cobjectric.specs import BooleanSpec

    assert BooleanSpec is not None


def test_boolean_spec_conversion() -> None:
    """Test BooleanSpec converts various values to bool."""

    from cobjectric.specs import BooleanSpec

    class Person(BaseModel):
        is_active: bool = BooleanSpec()

    person1 = Person(is_active=True)
    assert person1.fields.is_active.value is True

    person2 = Person(is_active=False)
    assert person2.fields.is_active.value is False

    person3 = Person(is_active=1)
    assert person3.fields.is_active.value is True

    person4 = Person(is_active=0)
    assert person4.fields.is_active.value is False


def test_boolean_spec_none_value() -> None:
    """Test BooleanSpec with None value."""

    from cobjectric.specs import BooleanSpec

    class Person(BaseModel):
        is_active: bool | None = BooleanSpec()

    person = Person(is_active=None)
    assert person.fields.is_active.value is None


def test_boolean_spec_exact_similarity() -> None:
    """Test that BooleanSpec uses exact similarity."""

    from cobjectric.specs import BooleanSpec

    class Person(BaseModel):
        is_active: bool = BooleanSpec()

    person_got = Person(is_active=True)
    person_expected = Person(is_active=True)

    result = person_got.compute_similarity(person_expected)
    assert result.fields.is_active.value == 1.0

    person_expected2 = Person(is_active=False)
    result2 = person_got.compute_similarity(person_expected2)
    assert result2.fields.is_active.value == 0.0


def test_datetime_spec_import() -> None:
    """Test that DatetimeSpec can be imported."""
    from cobjectric.specs import DatetimeSpec

    assert DatetimeSpec is not None


def test_datetime_spec_basic() -> None:
    """Test DatetimeSpec with ISO format string."""

    from cobjectric.specs import DatetimeSpec

    class Event(BaseModel):
        created_at: str = DatetimeSpec()

    event = Event(created_at="2024-01-15T10:30:00Z")
    assert event.fields.created_at.value == "2024-01-15T10:30:00Z"


def test_datetime_spec_normalize_datetime_object() -> None:
    """Test DatetimeSpec normalizes datetime objects to ISO format."""

    from datetime import datetime

    from cobjectric.specs import DatetimeSpec

    class Event(BaseModel):
        created_at: str = DatetimeSpec()

    dt = datetime(2024, 1, 15, 10, 30, 0)
    event = Event(created_at=dt)
    assert event.fields.created_at.value == "2024-01-15T10:30:00"


def test_datetime_spec_custom_format() -> None:
    """Test DatetimeSpec with custom format."""

    from cobjectric.specs import DatetimeSpec

    class Event(BaseModel):
        created_at: str = DatetimeSpec(format="%Y-%m-%d %H:%M:%S")

    event = Event(created_at="2024-01-15 10:30:00")
    assert event.fields.created_at.value == "2024-01-15T10:30:00"


def test_datetime_spec_auto_detect_formats() -> None:
    """Test DatetimeSpec auto-detects various ISO formats."""

    from cobjectric.specs import DatetimeSpec

    class Event(BaseModel):
        created_at: str = DatetimeSpec()

    # ISO format without Z
    event1 = Event(created_at="2024-01-15T10:30:00")
    assert event1.fields.created_at.value == "2024-01-15T10:30:00"

    # ISO format with Z
    event2 = Event(created_at="2024-01-15T10:30:00Z")
    assert event2.fields.created_at.value == "2024-01-15T10:30:00Z"

    # Space-separated format
    event3 = Event(created_at="2024-01-15 10:30:00")
    assert event3.fields.created_at.value == "2024-01-15T10:30:00"

    # Date only
    event4 = Event(created_at="2024-01-15")
    assert event4.fields.created_at.value == "2024-01-15T00:00:00"

    # Formats with "/"
    event5 = Event(created_at="2024/01/15 10:30:00")
    assert event5.fields.created_at.value == "2024-01-15T10:30:00"

    event6 = Event(created_at="2024/01/15")
    assert event6.fields.created_at.value == "2024-01-15T00:00:00"

    event7 = Event(created_at="15/01/2024 10:30:00")
    assert event7.fields.created_at.value == "2024-01-15T10:30:00"

    event8 = Event(created_at="15/01/2024")
    assert event8.fields.created_at.value == "2024-01-15T00:00:00"

    event9 = Event(created_at="01/15/2024 10:30:00")
    assert event9.fields.created_at.value == "2024-01-15T10:30:00"

    event10 = Event(created_at="01/15/2024")
    assert event10.fields.created_at.value == "2024-01-15T00:00:00"


def test_datetime_spec_invalid_format() -> None:
    """Test DatetimeSpec with invalid format."""

    from cobjectric.specs import DatetimeSpec

    class Event(BaseModel):
        created_at: str = DatetimeSpec(format="%Y-%m-%d %H:%M:%S")

    # Invalid format should return None (will fail type validation)
    event = Event(created_at="invalid")
    assert event.fields.created_at.value is MissingValue


def test_datetime_spec_non_string_value() -> None:
    """Test DatetimeSpec with non-string, non-datetime value."""

    from cobjectric.specs import DatetimeSpec

    class Event(BaseModel):
        created_at: str = DatetimeSpec()

    # Non-string, non-datetime value should return None (will fail type validation)
    event = Event(created_at=123)
    assert event.fields.created_at.value is MissingValue


def test_datetime_spec_parse_with_timezone() -> None:
    """Test DatetimeSpec parsing with timezone info."""

    from cobjectric.specs import DatetimeSpec

    class Event(BaseModel):
        created_at: str = DatetimeSpec()

    # ISO format with timezone offset (with colon)
    event1 = Event(created_at="2024-01-15T10:30:00+05:00")
    assert event1.fields.created_at.value == "2024-01-15T10:30:00+05:00"

    # ISO format with timezone offset (without colon)
    event2 = Event(created_at="2024-01-15T10:30:00+0500")
    assert event2.fields.created_at.value == "2024-01-15T10:30:00+05:00"

    # ISO format with negative timezone offset (with colon)
    event3 = Event(created_at="2024-01-15T10:30:00-03:00")
    assert event3.fields.created_at.value == "2024-01-15T10:30:00-03:00"

    # ISO format with negative timezone offset (without colon)
    event4 = Event(created_at="2024-01-15T10:30:00-0300")
    assert event4.fields.created_at.value == "2024-01-15T10:30:00-03:00"

    # Space-separated format with timezone
    event5 = Event(created_at="2024-01-15 10:30:00+05:00")
    assert event5.fields.created_at.value == "2024-01-15T10:30:00+05:00"

    # Format with "/" and timezone
    event6 = Event(created_at="2024/01/15 10:30:00+05:00")
    assert event6.fields.created_at.value == "2024-01-15T10:30:00+05:00"

    # Format with "/" and timezone without colon (needs normalization)
    event7 = Event(created_at="2024/01/15 10:30:00+0500")
    assert event7.fields.created_at.value == "2024-01-15T10:30:00+05:00"

    # Format with "/" and negative timezone without colon
    event8 = Event(created_at="15/01/2024 10:30:00-0300")
    assert event8.fields.created_at.value == "2024-01-15T10:30:00-03:00"


def test_datetime_spec_normalize_none_value() -> None:
    """Test DatetimeSpec normalizer with None value."""

    from cobjectric.specs import _normalize_datetime_to_iso

    result = _normalize_datetime_to_iso(None)
    assert result is None


def test_datetime_spec_normalize_datetime_object_direct() -> None:
    """Test DatetimeSpec normalizer with datetime object directly."""

    from datetime import datetime

    from cobjectric.specs import _normalize_datetime_to_iso

    dt = datetime(2024, 1, 15, 10, 30, 0)
    result = _normalize_datetime_to_iso(dt)
    assert result == "2024-01-15T10:30:00"


def test_datetime_spec_normalize_non_string_non_datetime() -> None:
    """Test DatetimeSpec normalizer with non-string, non-datetime value."""

    from cobjectric.specs import _normalize_datetime_to_iso

    result = _normalize_datetime_to_iso(123)
    assert result is None


def test_datetime_spec_custom_format_success() -> None:
    """Test DatetimeSpec with custom format that succeeds."""

    from cobjectric.specs import DatetimeSpec

    class Event(BaseModel):
        created_at: str = DatetimeSpec(format="%d/%m/%Y %H:%M")

    event = Event(created_at="15/01/2024 10:30")
    assert event.fields.created_at.value == "2024-01-15T10:30:00"


def test_datetime_spec_preserve_z_suffix_strptime() -> None:
    """Test DatetimeSpec preserves Z suffix when parsing with strptime (not fromisoformat)."""

    from cobjectric.specs import DatetimeSpec, _normalize_datetime_to_iso

    result = _normalize_datetime_to_iso("2024-01-15T10:30:00Z", format_str=None)
    assert result == "2024-01-15T10:30:00Z"

    class Event(BaseModel):
        created_at: str = DatetimeSpec(format="%Y-%m-%dT%H:%M:%SZ")

    event = Event(created_at="2024-01-15T10:30:00Z")
    assert event.fields.created_at.value == "2024-01-15T10:30:00Z"


def test_datetime_spec_timezone_normalization_success() -> None:
    """Test DatetimeSpec timezone normalization when strptime succeeds after normalization."""

    from cobjectric.specs import DatetimeSpec, _normalize_datetime_to_iso

    class Event(BaseModel):
        created_at: str = DatetimeSpec()

    event1 = Event(created_at="2024-01-15 10:30:00+0500")
    assert event1.fields.created_at.value == "2024-01-15T10:30:00+05:00"

    event2 = Event(created_at="2024/01/15 10:30:00+0500")
    assert event2.fields.created_at.value == "2024-01-15T10:30:00+05:00"

    event3 = Event(created_at="15/01/2024 10:30:00+0500")
    assert event3.fields.created_at.value == "2024-01-15T10:30:00+05:00"

    event4 = Event(created_at="01/15/2024 10:30:00+0500")
    assert event4.fields.created_at.value == "2024-01-15T10:30:00+05:00"

    result = _normalize_datetime_to_iso("2024/01/15 10:30:00+0500", format_str=None)
    assert result == "2024-01-15T10:30:00+05:00"


def test_datetime_spec_invalid_value_no_match() -> None:
    """Test DatetimeSpec with value that doesn't match any format."""

    from cobjectric.specs import DatetimeSpec

    class Event(BaseModel):
        created_at: str = DatetimeSpec()

    # Value that doesn't match any format should return None (will fail type validation)
    event = Event(created_at="not a date at all")
    assert event.fields.created_at.value is MissingValue


def test_datetime_spec_similarity_exact() -> None:
    """Test DatetimeSpec similarity with exact match."""

    from cobjectric.specs import DatetimeSpec

    class Event(BaseModel):
        created_at: str = DatetimeSpec()

    event_got = Event(created_at="2024-01-15T10:30:00Z")
    event_expected = Event(created_at="2024-01-15T10:30:00Z")

    result = event_got.compute_similarity(event_expected)
    assert result.fields.created_at.value == 1.0


def test_datetime_spec_similarity_with_tolerance() -> None:
    """Test DatetimeSpec similarity with max_difference."""

    from datetime import timedelta

    from cobjectric.specs import DatetimeSpec

    class Event(BaseModel):
        created_at: str = DatetimeSpec(max_difference=timedelta(hours=1))

    # 30 minutes difference
    event_got = Event(created_at="2024-01-15T10:00:00Z")
    event_expected = Event(created_at="2024-01-15T10:30:00Z")

    result = event_got.compute_similarity(event_expected)
    # diff=30min, 30min/60min=0.5, 1-0.5=0.5
    assert result.fields.created_at.value == pytest.approx(0.5)


def test_specs_combined_usage() -> None:
    """Test using multiple Specs together."""

    from cobjectric.specs import BooleanSpec, KeywordSpec, NumericSpec, TextSpec

    class Person(BaseModel):
        id: str = KeywordSpec()
        name: str = TextSpec()
        age: int = NumericSpec()
        is_active: bool = BooleanSpec()

    person = Person.from_dict(
        {
            "id": "  ABC123  ",
            "name": "  JOHN DOE  ",
            "age": 30.0,
            "is_active": True,
        }
    )

    assert person.fields.id.value == "ABC123"
    assert person.fields.name.value == "john doe"
    assert person.fields.age.value == 30
    assert isinstance(person.fields.age.value, int)
    assert person.fields.is_active.value is True


def test_specs_with_custom_parameters() -> None:
    """Test Specs with custom Spec parameters."""

    from cobjectric.specs import NumericSpec, TextSpec

    class Person(BaseModel):
        name: str = TextSpec(
            lower=False,
            similarity_weight=2.0,
            metadata={"description": "Full name"},
        )
        age: int = NumericSpec(
            max_difference=1.0,
            fill_rate_weight=1.5,
        )

    person = Person(name="JOHN", age=30.0)
    assert person.fields.name.value == "JOHN"
    assert person.fields.name.spec.metadata == {"description": "Full name"}
    assert person.fields.age.value == 30

    result = person.compute_similarity(Person(name="JOHN", age=30))
    assert result.fields.name.weight == 2.0

    fill_rate_result = person.compute_fill_rate()
    assert fill_rate_result.fields.age.weight == 1.5


def test_keyword_spec_non_string_value() -> None:
    """Test KeywordSpec with non-string value (non-int)."""

    from cobjectric.specs import KeywordSpec

    class Person(BaseModel):
        id: str = KeywordSpec()

    # With convert_int_to_str=True (default), int is converted to string
    person1 = Person(id=123)
    assert person1.fields.id.value == "123"

    # Non-string, non-int value should pass through unchanged (will fail type validation)
    person2 = Person(id=123.5)
    assert person2.fields.id.value is MissingValue

    # With convert_int_to_str=False, int stays as int (will fail type validation)
    class Person2(BaseModel):
        id: str = KeywordSpec(convert_int_to_str=False)

    person3 = Person2(id=123)
    assert person3.fields.id.value is MissingValue


def test_text_spec_non_string_value() -> None:
    """Test TextSpec with non-string value."""

    from cobjectric.specs import TextSpec

    class Person(BaseModel):
        name: str = TextSpec()

    # Non-string value should pass through unchanged (will fail type validation)
    person = Person(name=123)
    assert person.fields.name.value is MissingValue


def test_numeric_spec_conversion_error() -> None:
    """Test NumericSpec with value that can't be converted."""

    from cobjectric.specs import NumericSpec

    class Person(BaseModel):
        age: int = NumericSpec()

    # Value that can't be converted to int should return original value
    # (will fail type validation)
    person = Person(age="not a number")
    assert person.fields.age.value is MissingValue


def test_numeric_spec_non_numeric_type() -> None:
    """Test NumericSpec with non-numeric field type."""

    from cobjectric.specs import NumericSpec

    class Person(BaseModel):
        name: str = NumericSpec()  # Wrong: NumericSpec on str field

    # NumericSpec normalizer should not convert non-numeric types
    person = Person(name="John")
    assert person.fields.name.value == "John"


def test_numeric_spec_union_without_numeric() -> None:
    """Test NumericSpec with Union type that doesn't contain int/float."""

    from cobjectric.specs import NumericSpec

    class Person(BaseModel):
        value: str | None = NumericSpec()  # Union without numeric type

    # Normalizer should not convert (no numeric type found)
    person = Person(value="test")
    assert person.fields.value.value == "test"


def test_datetime_similarity_invalid_max_difference() -> None:
    """Test datetime_similarity_factory with invalid max_difference."""

    from datetime import timedelta

    from cobjectric.similarities import datetime_similarity_factory

    with pytest.raises(ValueError, match="max_difference must be > 0"):
        datetime_similarity_factory(max_difference=timedelta(seconds=0))

    with pytest.raises(ValueError, match="max_difference must be > 0"):
        datetime_similarity_factory(max_difference=timedelta(seconds=-1))


def test_datetime_similarity_none_values() -> None:
    """Test datetime_similarity with None values."""

    from cobjectric.similarities import datetime_similarity_factory

    similarity = datetime_similarity_factory()
    assert similarity(None, "2024-01-15T10:30:00Z") == 0.0
    assert similarity("2024-01-15T10:30:00Z", None) == 0.0
    assert similarity(None, None) == 0.0


def test_datetime_similarity_non_string_values() -> None:
    """Test datetime_similarity with non-string values."""

    from cobjectric.similarities import datetime_similarity_factory

    similarity = datetime_similarity_factory()
    assert similarity(123, "2024-01-15T10:30:00Z") == 0.0
    assert similarity("2024-01-15T10:30:00Z", 123) == 0.0


def test_datetime_similarity_invalid_format() -> None:
    """Test datetime_similarity with invalid datetime format."""

    from cobjectric.similarities import datetime_similarity_factory

    similarity = datetime_similarity_factory()
    assert similarity("invalid", "2024-01-15T10:30:00Z") == 0.0
    assert similarity("2024-01-15T10:30:00Z", "invalid") == 0.0


def test_datetime_similarity_datetime_object() -> None:
    """Test datetime_similarity with datetime objects."""

    from datetime import datetime

    from cobjectric.similarities import datetime_similarity_factory

    similarity = datetime_similarity_factory()
    dt1 = datetime(2024, 1, 15, 10, 30, 0)
    dt2 = datetime(2024, 1, 15, 10, 30, 0)
    assert similarity(dt1, dt2) == 1.0

    dt3 = datetime(2024, 1, 15, 11, 30, 0)
    assert similarity(dt1, dt3) == 0.0


def test_datetime_similarity_parse_datetime_none() -> None:
    """Test that parse_datetime handles None correctly."""

    from cobjectric.similarities import datetime_similarity_factory

    similarity = datetime_similarity_factory()
    # This should call parse_datetime with None, which returns None
    result = similarity(None, "2024-01-15T10:30:00Z")
    assert result == 0.0  # Because dt_a is None
