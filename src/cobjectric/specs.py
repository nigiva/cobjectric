import re
import types
import typing as t
import unicodedata
from datetime import datetime, timedelta

from cobjectric.context import FieldContext
from cobjectric.field_spec import Spec
from cobjectric.similarities import (
    datetime_similarity_factory,
    exact_similarity,
    fuzzy_similarity_factory,
    numeric_similarity_factory,
)


def _normalize_text(
    value: str,
    lower: bool,
    strip: bool,
    collapse_spaces: bool,
    remove_accents: bool,
) -> str:
    """
    Normalize text string.

    Args:
        value: The string to normalize.
        lower: If True, convert to lowercase.
        strip: If True, strip leading/trailing whitespace.
        collapse_spaces: If True, collapse multiple spaces to single space.
        remove_accents: If True, remove accents from characters.

    Returns:
        Normalized string.
    """
    result = value

    if remove_accents:
        # Remove accents using NFD decomposition and filtering
        result = "".join(
            c
            for c in unicodedata.normalize("NFD", result)
            if unicodedata.category(c) != "Mn"
        )

    if collapse_spaces:
        # Replace multiple spaces with single space
        # If strip=False, we need to preserve leading/trailing spaces
        if strip:
            # If we're going to strip anyway, collapse all spaces
            result = re.sub(r"\s+", " ", result)
        else:
            # Preserve leading and trailing spaces
            leading_match = re.match(r"^(\s*)", result)
            trailing_match = re.search(r"(\s*)$", result)
            leading = leading_match.group(1) if leading_match else ""
            trailing = trailing_match.group(1) if trailing_match else ""
            # Collapse only internal spaces
            internal = result[len(leading) : len(result) - len(trailing)]
            internal = re.sub(r"\s+", " ", internal)
            result = leading + internal + trailing

    if strip:
        result = result.strip()

    if lower:
        result = result.lower()

    return result


def _extract_numeric_type(field_type: type) -> type | None:
    """
    Extract the base numeric type from a field type.

    Handles Union types (e.g., int | None -> int) and direct types.

    Args:
        field_type: The field type to extract from.

    Returns:
        The base numeric type (int or float) or None if not numeric.
    """
    origin = t.get_origin(field_type)

    # Handle Union types
    if origin is t.Union or origin is types.UnionType:
        args = t.get_args(field_type)
        # Find int or float in the union
        for arg in args:
            if arg is int or arg is float:
                return arg
        return None

    # Direct type
    if field_type is int or field_type is float:
        return field_type

    return None


def KeywordSpec(  # noqa: N802
    strip: bool = True,
    convert_int_to_str: bool = True,
    metadata: dict[str, t.Any] | None = None,
    fill_rate_func: t.Any = None,
    fill_rate_weight: float = 1.0,
    fill_rate_accuracy_func: t.Any = None,
    fill_rate_accuracy_weight: float = 1.0,
    similarity_func: t.Any = None,
    similarity_weight: float = 1.0,
    list_compare_strategy: t.Any = None,
) -> t.Any:
    """
    Pre-defined Spec for keyword/identifier fields (exact matching).

    Optimized for identifiers, codes, enums, etc. Uses exact similarity
    and optional string stripping and int-to-string conversion.

    Args:
        strip: If True (default), strip leading/trailing whitespace.
        convert_int_to_str: If True (default), convert int values to string.
            Useful when IDs come as integers from JSON but should be
            compared as strings.
        metadata: Optional metadata for the field.
        fill_rate_func: Optional fill rate function.
        fill_rate_weight: Weight for fill rate computation (default: 1.0).
        fill_rate_accuracy_func: Optional fill rate accuracy function.
        fill_rate_accuracy_weight: Weight for fill rate accuracy (default: 1.0).
        similarity_func: Optional similarity function (default: exact_similarity).
        similarity_weight: Weight for similarity computation (default: 1.0).
        list_compare_strategy: Strategy for comparing list[BaseModel] items.

    Returns:
        FieldSpec instance optimized for keyword fields.

    Example:
        ```python
        class Person(BaseModel):
            id: str = KeywordSpec()  # Converts 123 -> "123"
        ```
    """
    normalizer = None
    if strip or convert_int_to_str:

        def keyword_normalizer(value: t.Any) -> t.Any:
            # Convert int to string if enabled
            if convert_int_to_str and isinstance(value, int):
                value = str(value)
            # Strip whitespace if enabled
            if strip and isinstance(value, str):
                return value.strip()
            return value

        normalizer = keyword_normalizer

    return Spec(
        metadata=metadata,
        normalizer=normalizer,
        fill_rate_func=fill_rate_func,
        fill_rate_weight=fill_rate_weight,
        fill_rate_accuracy_func=fill_rate_accuracy_func,
        fill_rate_accuracy_weight=fill_rate_accuracy_weight,
        similarity_func=(
            similarity_func if similarity_func is not None else exact_similarity
        ),
        similarity_weight=similarity_weight,
        list_compare_strategy=list_compare_strategy,
    )


def TextSpec(  # noqa: N802
    lower: bool = True,
    strip: bool = True,
    collapse_spaces: bool = True,
    remove_accents: bool = True,
    scorer: str = "WRatio",
    metadata: dict[str, t.Any] | None = None,
    fill_rate_func: t.Any = None,
    fill_rate_weight: float = 1.0,
    fill_rate_accuracy_func: t.Any = None,
    fill_rate_accuracy_weight: float = 1.0,
    similarity_func: t.Any = None,
    similarity_weight: float = 1.0,
    list_compare_strategy: t.Any = None,
) -> t.Any:
    """
    Pre-defined Spec for text fields (fuzzy matching).

    Optimized for free-form text. Uses fuzzy similarity and comprehensive
    text normalization (lowercase, strip, collapse spaces, remove accents).

    Args:
        lower: If True (default), convert to lowercase.
        strip: If True (default), strip leading/trailing whitespace.
        collapse_spaces: If True (default), collapse multiple spaces to single.
        remove_accents: If True (default), remove accents from characters.
        scorer: The rapidfuzz scorer to use (default: "WRatio").
        metadata: Optional metadata for the field.
        fill_rate_func: Optional fill rate function.
        fill_rate_weight: Weight for fill rate computation (default: 1.0).
        fill_rate_accuracy_func: Optional fill rate accuracy function.
        fill_rate_accuracy_weight: Weight for fill rate accuracy (default: 1.0).
        similarity_func: Optional similarity function (default: fuzzy).
        similarity_weight: Weight for similarity computation (default: 1.0).
        list_compare_strategy: Strategy for comparing list[BaseModel] items.

    Returns:
        FieldSpec instance optimized for text fields.

    Example:
        ```python
        class Person(BaseModel):
            name: str = TextSpec()
        ```
    """
    normalizer = None
    if lower or strip or collapse_spaces or remove_accents:

        def text_normalizer(value: t.Any) -> t.Any:
            if isinstance(value, str):
                return _normalize_text(
                    value,
                    lower=lower,
                    strip=strip,
                    collapse_spaces=collapse_spaces,
                    remove_accents=remove_accents,
                )
            return value

        normalizer = text_normalizer

    return Spec(
        metadata=metadata,
        normalizer=normalizer,
        fill_rate_func=fill_rate_func,
        fill_rate_weight=fill_rate_weight,
        fill_rate_accuracy_func=fill_rate_accuracy_func,
        fill_rate_accuracy_weight=fill_rate_accuracy_weight,
        similarity_func=(
            similarity_func
            if similarity_func is not None
            else fuzzy_similarity_factory(scorer=scorer)
        ),
        similarity_weight=similarity_weight,
        list_compare_strategy=list_compare_strategy,
    )


def NumericSpec(  # noqa: N802
    max_difference: float | None = None,
    coerce_type: bool = True,
    metadata: dict[str, t.Any] | None = None,
    fill_rate_func: t.Any = None,
    fill_rate_weight: float = 1.0,
    fill_rate_accuracy_func: t.Any = None,
    fill_rate_accuracy_weight: float = 1.0,
    similarity_func: t.Any = None,
    similarity_weight: float = 1.0,
    list_compare_strategy: t.Any = None,
) -> t.Any:
    """
    Pre-defined Spec for numeric fields.

    Optimized for numbers (int, float). Uses numeric similarity with optional
    tolerance. Can coerce JSON Number to int/float based on field type.

    Args:
        max_difference: Maximum difference for similarity (None = exact match).
        coerce_type: If True (default), coerce to int/float based on field type.
            Handles JSON Number -> Python int/float conversion.
        metadata: Optional metadata for the field.
        fill_rate_func: Optional fill rate function.
        fill_rate_weight: Weight for fill rate computation (default: 1.0).
        fill_rate_accuracy_func: Optional fill rate accuracy function.
        fill_rate_accuracy_weight: Weight for fill rate accuracy (default: 1.0).
        similarity_func: Optional similarity function (default: numeric).
        similarity_weight: Weight for similarity computation (default: 1.0).
        list_compare_strategy: Strategy for comparing list[BaseModel] items.

    Returns:
        FieldSpec instance optimized for numeric fields.

    Example:
        ```python
        class Person(BaseModel):
            age: int = NumericSpec()  # 30.0 from JSON -> 30 (int)
            score: float = NumericSpec(max_difference=0.5)
        ```
    """
    normalizer = None
    if coerce_type:

        def numeric_normalizer(value: t.Any, context: FieldContext) -> t.Any:
            if value is None:
                return value

            try:
                base_type = _extract_numeric_type(context.field_type)
                if base_type is int:
                    return int(float(value))  # 10.0 -> 10
                elif base_type is float:
                    return float(value)  # 10 -> 10.0
            except (ValueError, TypeError):
                pass

            return value

        normalizer = numeric_normalizer

    return Spec(
        metadata=metadata,
        normalizer=normalizer,
        fill_rate_func=fill_rate_func,
        fill_rate_weight=fill_rate_weight,
        fill_rate_accuracy_func=fill_rate_accuracy_func,
        fill_rate_accuracy_weight=fill_rate_accuracy_weight,
        similarity_func=(
            similarity_func
            if similarity_func is not None
            else numeric_similarity_factory(max_difference=max_difference)
        ),
        similarity_weight=similarity_weight,
        list_compare_strategy=list_compare_strategy,
    )


def BooleanSpec(  # noqa: N802
    metadata: dict[str, t.Any] | None = None,
    fill_rate_func: t.Any = None,
    fill_rate_weight: float = 1.0,
    fill_rate_accuracy_func: t.Any = None,
    fill_rate_accuracy_weight: float = 1.0,
    similarity_func: t.Any = None,
    similarity_weight: float = 1.0,
    list_compare_strategy: t.Any = None,
) -> t.Any:
    """
    Pre-defined Spec for boolean fields.

    Optimized for boolean values. Uses exact similarity and converts
    various values to bool.

    Args:
        metadata: Optional metadata for the field.
        fill_rate_func: Optional fill rate function.
        fill_rate_weight: Weight for fill rate computation (default: 1.0).
        fill_rate_accuracy_func: Optional fill rate accuracy function.
        fill_rate_accuracy_weight: Weight for fill rate accuracy (default: 1.0).
        similarity_func: Optional similarity function (default: exact_similarity).
        similarity_weight: Weight for similarity computation (default: 1.0).
        list_compare_strategy: Strategy for comparing list[BaseModel] items.

    Returns:
        FieldSpec instance optimized for boolean fields.

    Example:
        ```python
        class Person(BaseModel):
            is_active: bool = BooleanSpec()
        ```
    """

    def boolean_normalizer(value: t.Any) -> t.Any:
        if value is None:
            return None
        return bool(value)

    return Spec(
        metadata=metadata,
        normalizer=boolean_normalizer,
        fill_rate_func=fill_rate_func,
        fill_rate_weight=fill_rate_weight,
        fill_rate_accuracy_func=fill_rate_accuracy_func,
        fill_rate_accuracy_weight=fill_rate_accuracy_weight,
        similarity_func=(
            similarity_func if similarity_func is not None else exact_similarity
        ),
        similarity_weight=similarity_weight,
        list_compare_strategy=list_compare_strategy,
    )


def _normalize_datetime_to_iso(
    value: t.Any, format_str: str | None = None
) -> str | None:
    """
    Normalize datetime value to ISO format string.

    Args:
        value: The value to normalize (datetime, str, or None).
        format_str: Optional format string for parsing (auto-detect if None).

    Returns:
        ISO format string or None if parsing fails.
    """
    if value is None:
        return None

    # If already a datetime object, convert to ISO format
    if isinstance(value, datetime):
        return value.isoformat()

    # If not a string, return None (will fail type validation)
    if not isinstance(value, str):
        return None

    # First, try to parse with fromisoformat if it looks like ISO format
    # This preserves Z suffix and timezone info
    try:
        # Handle Z suffix (UTC)
        if value.endswith("Z"):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.isoformat().replace("+00:00", "Z")
        # Handle timezone offsets like +05:00, -03:00, +0500, -0300
        elif "+" in value[-6:] or (value.count("-") > 2 and "T" in value):
            # Try parsing with fromisoformat (handles +05:00, -03:00, etc.)
            dt = datetime.fromisoformat(value)
            return dt.isoformat()
        else:
            dt = datetime.fromisoformat(value)
            return dt.isoformat()
    except (ValueError, AttributeError):
        pass

    # If format is specified, use it
    if format_str is not None:
        try:
            dt = datetime.strptime(value, format_str)
            return dt.isoformat()
        except ValueError:
            return None

    # Otherwise, try common ISO formats (auto-detect)
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",  # +05:00 or -03:00
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        # Formats with "/"
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
        # Formats with timezone
        "%Y-%m-%d %H:%M:%S%z",
        "%Y/%m/%d %H:%M:%S%z",
        "%d/%m/%Y %H:%M:%S%z",
        "%m/%d/%Y %H:%M:%S%z",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).isoformat()
        except ValueError:
            continue

    return None


def DatetimeSpec(  # noqa: N802
    format: str | None = None,
    max_difference: timedelta | None = None,
    metadata: dict[str, t.Any] | None = None,
    fill_rate_func: t.Any = None,
    fill_rate_weight: float = 1.0,
    fill_rate_accuracy_func: t.Any = None,
    fill_rate_accuracy_weight: float = 1.0,
    similarity_func: t.Any = None,
    similarity_weight: float = 1.0,
    list_compare_strategy: t.Any = None,
) -> t.Any:
    """
    Pre-defined Spec for datetime fields.

    Optimized for datetime strings (ISO format). Uses datetime similarity
    with optional time difference tolerance. Normalizes all datetime values
    to ISO format strings.

    Args:
        format: Optional datetime format string for parsing (auto-detect if None).
            If specified, uses datetime.strptime() with this format.
            Example: "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", etc.
        max_difference: Maximum time difference as timedelta (None = exact).
            Use timedelta(days=1), timedelta(hours=1), timedelta(minutes=30), etc.
        metadata: Optional metadata for the field.
        fill_rate_func: Optional fill rate function.
        fill_rate_weight: Weight for fill rate computation (default: 1.0).
        fill_rate_accuracy_func: Optional fill rate accuracy function.
        fill_rate_accuracy_weight: Weight for fill rate accuracy (default: 1.0).
        similarity_func: Optional similarity function (default: datetime).
        similarity_weight: Weight for similarity computation (default: 1.0).
        list_compare_strategy: Strategy for comparing list[BaseModel] items.

    Returns:
        FieldSpec instance optimized for datetime fields.

    Example:
        ```python
        from datetime import timedelta

        class Event(BaseModel):
            # Auto-detect format
            created_at: str = DatetimeSpec(max_difference=timedelta(hours=1))

            # Custom format
            updated_at: str = DatetimeSpec(format="%Y-%m-%d %H:%M:%S")
        ```
    """

    def datetime_normalizer(value: t.Any) -> t.Any:
        return _normalize_datetime_to_iso(value, format_str=format)

    return Spec(
        metadata=metadata,
        normalizer=datetime_normalizer,
        fill_rate_func=fill_rate_func,
        fill_rate_weight=fill_rate_weight,
        fill_rate_accuracy_func=fill_rate_accuracy_func,
        fill_rate_accuracy_weight=fill_rate_accuracy_weight,
        similarity_func=(
            similarity_func
            if similarity_func is not None
            else datetime_similarity_factory(max_difference=max_difference)
        ),
        similarity_weight=similarity_weight,
        list_compare_strategy=list_compare_strategy,
    )
