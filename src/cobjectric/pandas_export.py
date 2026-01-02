from __future__ import annotations

import typing as t

from cobjectric.results import (
    AggregatedFieldResult,
    AggregatedModelResult,
    FieldResult,
    ListResult,
    ModelResult,
)


def _check_pandas_available() -> bool:
    """
    Check if pandas is available.

    Returns:
        True if pandas is installed, False otherwise.
    """
    try:
        import pandas  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _get_pandas() -> t.Any:
    """
    Import pandas if available.

    Returns:
        pandas module.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas  # noqa: PLC0415

        return pandas
    except ImportError as e:
        raise ImportError(
            "pandas is required for to_series() and to_dataframe(). "
            "Install it with: pip install cobjectric[pandas]"
        ) from e


def _flatten_to_dict(
    result: ModelResult | FieldResult | ListResult,
    prefix: str = "",
) -> dict[str, float]:
    """
    Flatten a ModelResult, FieldResult, or ListResult into a dictionary.

    For nested models, uses dot notation: address.city
    For lists, uses aggregated_fields and mean: experiences.title

    Args:
        result: The result to flatten.
        prefix: Prefix for field names (used for nested models).

    Returns:
        Dictionary mapping field paths to values.
    """
    flattened: dict[str, float] = {}

    if isinstance(result, FieldResult):
        # Simple field - just add the value
        if prefix:
            flattened[prefix] = result.value
        else:
            # This shouldn't happen in normal usage, but handle it
            flattened["value"] = result.value
    elif isinstance(result, ModelResult):
        # Model result - recursively flatten all fields
        for field_name, field_result in result._fields.items():
            field_path = f"{prefix}.{field_name}" if prefix else field_name
            nested_flattened = _flatten_to_dict(field_result, field_path)
            flattened.update(nested_flattened)
    elif isinstance(result, ListResult):
        # List result - use aggregated_fields to get mean values
        if not result._items:
            # Empty list - nothing to flatten
            return flattened

        aggregated = result.aggregated_fields
        for field_name in result._items[0]._fields.keys():
            field_path = f"{prefix}.{field_name}" if prefix else field_name
            aggregated_field = getattr(aggregated, field_name)

            if isinstance(aggregated_field, AggregatedFieldResult):
                # Simple aggregated field - use mean
                flattened[field_path] = aggregated_field.mean()
            elif isinstance(aggregated_field, AggregatedModelResult):
                # Nested model in list - recursively flatten using mean of each item
                # For each field in the nested model, calculate mean across items
                for nested_field_name in aggregated_field._items[0]._fields.keys():
                    nested_field_path = (
                        f"{field_path}.{nested_field_name}"
                        if field_path
                        else nested_field_name
                    )
                    # Get aggregated result for this nested field
                    nested_aggregated = getattr(aggregated_field, nested_field_name)
                    if isinstance(nested_aggregated, AggregatedFieldResult):
                        flattened[nested_field_path] = nested_aggregated.mean()
                    # For deeper nesting, we'd need to recurse further
            # For nested lists, we don't flatten further here
            # They would be accessed via aggregated_fields in the DataFrame

    return flattened
