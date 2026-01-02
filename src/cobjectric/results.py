from __future__ import annotations

import typing as t
from dataclasses import dataclass

from cobjectric.exceptions import InvalidAggregatedFieldError
from cobjectric.path import parse_path
from cobjectric.stats import StatsMixin


@dataclass
class FieldResult:
    """
    Result of fill rate computation for a single field.

    Attributes:
        value: The fill rate value (float between 0.0 and 1.0).
        weight: Weight for this field in weighted mean calculation (default: 1.0).
    """

    value: float
    weight: float = 1.0

    def __repr__(self) -> str:
        """Return a string representation of the FieldResult."""
        return f"FieldResult(value={self.value}, weight={self.weight})"


class FieldResultCollection:
    """
    Collection of fill rate results for a model instance.

    Provides attribute-based access to fill rate results.
    """

    def __init__(
        self,
        fields: dict[
            str,
            FieldResult | ModelResult | ListResult,
        ],
    ) -> None:
        """
        Initialize a FieldResultCollection.

        Args:
            fields: Dictionary mapping field names to FieldResult,
                ModelResult, or ListResult instances.
        """
        self._fields = fields

    def __getattr__(self, name: str) -> FieldResult | ModelResult | ListResult:
        """
        Get a fill rate result by field name.

        Args:
            name: The name of the field.

        Returns:
            The FieldResult or ModelResult instance.

        Raises:
            AttributeError: If the field does not exist.
        """
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __iter__(
        self,
    ) -> t.Iterator[FieldResult | ModelResult | ListResult]:
        """Iterate over all fill rate results."""
        return iter(self._fields.values())

    def __getitem__(self, path: str) -> FieldResult | ModelResult | ListResult:
        """
        Get a fill rate result by path.

        Args:
            path: Path to the field (e.g., "name", "address.city", "items[0].name").

        Returns:
            The FieldResult or ModelResult instance.

        Raises:
            KeyError: If the path is invalid.
        """
        segments = parse_path(path)
        return self._resolve_path(segments)

    def _resolve_path(
        self, segments: list[str]
    ) -> FieldResult | ModelResult | ListResult:
        """
        Resolve a path from segments.

        Args:
            segments: List of path segments.

        Returns:
            The FieldResult or ModelResult instance.

        Raises:
            KeyError: If the path is invalid.
        """
        if not segments:
            raise KeyError("Empty path")

        current: FieldResult | ModelResult | ListResult | None = None
        current_fields = self._fields

        for i, segment in enumerate(segments):
            if segment.startswith("[") and segment.endswith("]"):
                # List index access
                index_str = segment[1:-1]
                try:
                    index = int(index_str)
                except ValueError as e:
                    raise KeyError(f"Invalid list index: {index_str}") from e

                if isinstance(current, ListResult):
                    try:
                        current = current[index]
                        if i < len(segments) - 1:
                            current_fields = current._fields
                    except IndexError as e:
                        raise KeyError(f"List index {index} out of range") from e
                else:
                    raise KeyError(f"Cannot use index on non-list field: {segment}")
                continue

            if segment not in current_fields:
                raise KeyError(f"Field '{segment}' not found in path")

            current = current_fields[segment]

            if i < len(segments) - 1:
                # More segments to go
                next_segment = segments[i + 1]
                # Check if next segment is a list index
                if next_segment.startswith("[") and next_segment.endswith("]"):
                    # Will be handled in next iteration
                    if isinstance(current, ListResult):
                        # current_fields stays as is, updated when accessing index
                        continue
                    raise KeyError(f"Cannot use index on non-list field '{segment}'")

                # Current must be a ModelResult or ListResult
                if isinstance(current, ModelResult):
                    current_fields = current._fields
                elif isinstance(current, ListResult):
                    # For list, we need to access aggregated or wait for index
                    raise KeyError(
                        f"Cannot access '{next_segment}' directly on list field "
                        f"'{segment}'. Use index like '{segment}[0]' or aggregated "
                        f"access like '{segment}.{next_segment}'"
                    )
                else:
                    raise KeyError(
                        f"Cannot access '{next_segment}' "
                        f"on non-model field '{segment}'"
                    )

        assert current is not None, "Invalid path"
        return current

    def __repr__(self) -> str:
        """Return a string representation of the FieldResultCollection."""
        fields_repr = ", ".join(f"{name}=..." for name in self._fields.keys())
        return f"FieldResultCollection({fields_repr})"


@dataclass
class ModelResult(StatsMixin):
    """
    Result of fill rate computation for a model instance.

    Attributes:
        _fields: Dictionary mapping field names to FieldResult,
            ModelResult, or ListResult instances.
    """

    _fields: dict[str, FieldResult | ModelResult | ListResult]

    def _get_values(self) -> list[float]:
        """Get all values for statistical computation."""
        return self._collect_all_values()

    def _get_values_and_weights(
        self,
    ) -> tuple[list[float], list[float]]:
        """Get all values and weights for weighted statistical computation."""
        return self._collect_all_values_and_weights()

    @property
    def fields(self) -> FieldResultCollection:
        """
        Get the FieldResultCollection for this result.

        Returns:
            The FieldResultCollection containing all fill rate results.
        """
        return FieldResultCollection(self._fields)

    def __getitem__(self, path: str) -> FieldResult | ModelResult | ListResult:
        """
        Get a fill rate result by path.

        Args:
            path: Path to the field (e.g., "name", "address.city", "items[0].name").

        Returns:
            The FieldResult or ModelResult instance.

        Raises:
            KeyError: If the path is invalid.
        """
        return self.fields[path]

    def _collect_all_values(self) -> list[float]:
        """
        Collect all fill rate values recursively (including nested models).

        Returns:
            List of all fill rate values as floats.
        """
        values: list[float] = []
        for field_result in self._fields.values():
            if isinstance(field_result, ModelResult):
                values.extend(field_result._collect_all_values())
            elif isinstance(field_result, ListResult):
                for item in field_result._items:
                    values.extend(item._collect_all_values())
            else:
                values.append(field_result.value)
        return values

    def _collect_all_values_and_weights(
        self,
    ) -> tuple[list[float], list[float]]:
        """
        Collect all fill rate values and weights recursively (including nested models).

        Returns:
            Tuple of (values, weights) lists.
        """
        values: list[float] = []
        weights: list[float] = []
        for field_result in self._fields.values():
            if isinstance(field_result, ModelResult):
                nested_values, nested_weights = (
                    field_result._collect_all_values_and_weights()
                )
                values.extend(nested_values)
                weights.extend(nested_weights)
            elif isinstance(field_result, ListResult):
                for item in field_result._items:
                    item_values, item_weights = item._collect_all_values_and_weights()
                    values.extend(item_values)
                    weights.extend(item_weights)
            else:
                values.append(field_result.value)
                weights.append(field_result.weight)
        return values, weights

    def to_series(self) -> t.Any:
        """
        Export to pandas Series (requires cobjectric[pandas]).

        Returns:
            pandas.Series with field paths as index and values as data.

        Raises:
            ImportError: If pandas is not installed.
        """
        from cobjectric.pandas_export import (  # noqa: PLC0415
            _flatten_to_dict,
            _get_pandas,
        )

        pd = _get_pandas()
        flattened = _flatten_to_dict(self)
        return pd.Series(flattened)

    def __add__(self, other: "ModelResult") -> "ModelResultCollection":
        """
        Combine two ModelResults into a collection.

        Args:
            other: Another ModelResult to combine.

        Returns:
            ModelResultCollection containing both results.

        Raises:
            IncompatibleModelResultError: If results come from different model types.
        """
        from cobjectric.exceptions import IncompatibleModelResultError  # noqa: PLC0415

        # Check if both results come from the same model type
        # We can infer this by checking if they have the same field structure
        self_fields = set(self._fields.keys())
        other_fields = set(other._fields.keys())

        if self_fields != other_fields:
            # Try to get model types from the results if possible
            # For now, we'll use a generic error message
            raise IncompatibleModelResultError(type(self), type(other))

        return ModelResultCollection([self, other])

    def __repr__(self) -> str:
        """Return a string representation of the ModelResult."""
        fields_repr = ", ".join(
            f"{name}={field!r}" for name, field in self._fields.items()
        )
        return f"ModelResult({fields_repr})"


@dataclass
class AggregatedFieldResult(StatsMixin):
    """
    Aggregated fill rate result for a field across list items.

    Provides statistical methods (mean, std, etc.) over all values
    of a specific field across multiple items in a list.
    """

    _values: list[float]
    _weights: list[float]

    @property
    def values(self) -> list[float]:
        """
        Get all fill rate values for this field across items.

        Returns:
            List of fill rate values.
        """
        return self._values

    def _get_values(self) -> list[float]:
        """Get all values for statistical computation."""
        return self._values

    def _get_values_and_weights(
        self,
    ) -> tuple[list[float], list[float]]:
        """Get all values and weights for weighted statistical computation."""
        return self._values, self._weights

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"AggregatedFieldResult(values={self._values})"


@dataclass
class AggregatedModelResult(StatsMixin):
    """
    Aggregated result for a nested model field across list items.

    Allows chained access: items.aggregated_fields.address.city

    Note:
        For nested lists (list[list[BaseModel]]), accessing a nested list field
        returns the mean fill rate of each list, not individual field access.
        For example, if items have a tags field of type list[Tag], accessing
        items.aggregated_fields.tags returns a AggregatedFieldResult
        with the mean fill rate of each tags list, not access to individual
        tag fields.
    """

    _items: list[ModelResult]

    def _get_values(self) -> list[float]:
        """Get all values for statistical computation (mean of each item)."""
        return [item.mean() for item in self._items]

    def _get_values_and_weights(
        self,
    ) -> tuple[list[float], list[float]]:
        """
        Get all values and weights for weighted statistical computation.

        For aggregated model results, we use equal weights (1.0) for each item.
        """
        values = self._get_values()
        weights = [1.0] * len(values)
        return values, weights

    def __getattr__(
        self, name: str
    ) -> AggregatedFieldResult | AggregatedModelResult | NestedListAggregatedResult:
        """
        Get aggregated result for a field name across items.

        Args:
            name: The field name.

        Returns:
            AggregatedFieldResult, AggregatedModelResult,
            or NestedListAggregatedResult.

        Raises:
            InvalidAggregatedFieldError: If the field doesn't exist in the model.
        """
        # Validate field exists
        if self._items and name not in self._items[0]._fields:
            available = list(self._items[0]._fields.keys())
            raise InvalidAggregatedFieldError(name, available)

        values: list[float] = []
        weights: list[float] = []
        nested_items: list[ModelResult] = []
        nested_lists: list[ListResult] = []

        for item in self._items:
            if name in item._fields:
                field = item._fields[name]
                if isinstance(field, FieldResult):
                    values.append(field.value)
                    weights.append(field.weight)
                elif isinstance(field, ModelResult):
                    nested_items.append(field)
                elif isinstance(field, ListResult):
                    # Nested list - collect for NestedListAggregatedResult
                    nested_lists.append(field)

        # If we have nested lists, return NestedListAggregatedResult
        if nested_lists:
            element_type = nested_lists[0]._element_type if nested_lists else None
            return NestedListAggregatedResult(nested_lists, element_type)

        if nested_items:
            return AggregatedModelResult(_items=nested_items)
        return AggregatedFieldResult(_values=values, _weights=weights)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"AggregatedModelResult(items={len(self._items)})"


class NestedListAggregatedResult:
    """
    Aggregated result for nested lists in fill rate results.

    When accessing a list field through aggregated_fields, this allows
    chaining to further aggregate nested lists and access their fields.
    """

    def __init__(
        self, lists: list[ListResult], element_type: type | None = None
    ) -> None:
        """
        Initialize NestedListAggregatedResult.

        Args:
            lists: List of ListResult instances.
            element_type: The element type of the list (optional).
        """
        self._lists = lists
        self._element_type = element_type

    @property
    def aggregated_fields(self) -> AggregatedFieldResultCollection:
        """
        Get aggregated fields across all nested lists.

        Flattens the nested lists and returns a collection for accessing
        fields across all items.

        Returns:
            AggregatedFieldResultCollection instance.
        """
        # Flatten all items from all lists
        all_items: list[ModelResult] = []
        for list_result in self._lists:
            all_items.extend(list_result._items)
        return AggregatedFieldResultCollection(all_items)

    @property
    def values(self) -> list[float]:
        """
        Get fill rate values for nested lists.

        By default, returns the mean fill rate of each list (one value per list).
        This is the hierarchical view. To get flattened values, use
        NestedListAggregatedResult.aggregated_fields to access individual
        items and their fields.

        Returns:
            List of mean fill rates (one per list).
        """
        return [lst.mean() for lst in self._lists]

    def mean(self, hierarchical: bool = False) -> float:
        """
        Calculate mean fill rate across nested lists.

        Args:
            hierarchical: If False (default), flattens all values and calculates
                mean. If True, calculates mean of means for each list.

        Returns:
            Mean fill rate.
        """
        if not self._lists:
            return 0.0

        if hierarchical:
            # Mean of means: calculate mean for each list, then mean of those
            means = [lst.mean() for lst in self._lists]
            return sum(means) / len(means) if means else 0.0

        # Flatten: collect all values and calculate mean
        all_values, all_weights = self._collect_all_values_and_weights()
        if not all_values:
            return 0.0
        total_weight = sum(all_weights)
        if total_weight == 0.0:
            return 0.0
        return (
            sum(v * w for v, w in zip(all_values, all_weights, strict=True))
            / total_weight
        )

    def std(self, hierarchical: bool = False) -> float:
        """
        Calculate standard deviation across nested lists.

        Args:
            hierarchical: If False (default), calculates std of flattened values.
                If True, calculates std of means for each list.

        Returns:
            Standard deviation.
        """
        if len(self._lists) <= 1:
            return 0.0

        if hierarchical:
            # Std of means
            means = [lst.mean() for lst in self._lists]
            m = self.mean(hierarchical=True)
            return (sum((x - m) ** 2 for x in means) / len(means)) ** 0.5

        # Std of flattened values
        values = self.values
        assert len(values) > 1  # because list is not empty
        m = self.mean(hierarchical=False)
        return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5

    def var(self, hierarchical: bool = False) -> float:
        """
        Calculate variance across nested lists.

        Args:
            hierarchical: If False (default), calculates variance of flattened values.
                If True, calculates variance of means for each list.

        Returns:
            Variance.
        """
        if len(self._lists) <= 1:
            return 0.0

        if hierarchical:
            # Var of means
            means = [lst.mean() for lst in self._lists]
            m = self.mean(hierarchical=True)
            return sum((x - m) ** 2 for x in means) / len(means)

        # Var of flattened values
        values = self.values
        assert len(values) > 1  # because list is not empty
        m = self.mean(hierarchical=False)
        return sum((x - m) ** 2 for x in values) / len(values)

    def max(self, hierarchical: bool = False) -> float:
        """
        Get maximum fill rate.

        Args:
            hierarchical: If False (default), returns max of flattened values.
                If True, returns max of means for each list.

        Returns:
            Maximum fill rate.
        """
        if not self._lists:
            return 0.0

        if hierarchical:
            means = [lst.mean() for lst in self._lists]
            return max(means) if means else 0.0

        values = self.values
        return max(values) if values else 0.0

    def min(self, hierarchical: bool = False) -> float:
        """
        Get minimum fill rate.

        Args:
            hierarchical: If False (default), returns min of flattened values.
                If True, returns min of means for each list.

        Returns:
            Minimum fill rate.
        """
        if not self._lists:
            return 0.0

        if hierarchical:
            means = [lst.mean() for lst in self._lists]
            return min(means) if means else 0.0

        values = self.values
        return min(values) if values else 0.0

    def quantile(self, q: float, hierarchical: bool = False) -> float:
        """
        Calculate quantile of fill rates.

        Args:
            q: The quantile to compute (float between 0.0 and 1.0).
            hierarchical: If False (default), calculates quantile of flattened values.
                If True, calculates quantile of means for each list.

        Returns:
            The quantile value.

        Raises:
            ValueError: If q is not in [0, 1].
        """
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantile q must be between 0.0 and 1.0, got {q}")

        if not self._lists:
            return 0.0

        if hierarchical:
            means = [lst.mean() for lst in self._lists]
            values = sorted(means)
        else:
            values = sorted(self.values)

        assert len(values) > 0  # because list is not empty

        if q == 0.0:
            return values[0]
        if q == 1.0:
            return values[-1]

        n = len(values)
        index = q * (n - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, n - 1)
        weight = index - lower_index

        return values[lower_index] * (1 - weight) + values[upper_index] * weight

    def _collect_all_values_and_weights(self) -> tuple[list[float], list[float]]:
        """
        Collect all values and weights from nested lists (flattened).

        Returns:
            Tuple of (values, weights) lists.
        """
        values: list[float] = []
        weights: list[float] = []
        for list_result in self._lists:
            v, w = list_result._collect_all_values_and_weights()
            values.extend(v)
            weights.extend(w)
        return values, weights

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"NestedListAggregatedResult(lists={len(self._lists)})"


class AggregatedFieldResultCollection:
    """
    Collection for aggregated field access on a list of fill rate results.

    Provides access to aggregated field results across all items
    in the list.

    Note:
        For nested lists (list[list[BaseModel]]), accessing a nested list
        field returns the mean fill rate of each list, not individual field
        access. Use indexed access to access nested list items individually.
    """

    def __init__(self, items: list[ModelResult]) -> None:
        """
        Initialize the collection.

        Args:
            items: List of ModelResult instances.
        """
        self._items = items

    def _get_available_fields(self) -> list[str]:
        """Get list of available field names from first item."""
        if self._items:
            return list(self._items[0]._fields.keys())
        return []

    def __getattr__(
        self, name: str
    ) -> AggregatedFieldResult | AggregatedModelResult | NestedListAggregatedResult:
        """
        Get aggregated result for a field name across all items.

        Args:
            name: The field name.

        Returns:
            AggregatedFieldResult, AggregatedModelResult,
            or NestedListAggregatedResult.

        Raises:
            InvalidAggregatedFieldError: If the field doesn't exist in the model.
        """
        # Validate field exists
        if self._items and name not in self._items[0]._fields:
            available = self._get_available_fields()
            raise InvalidAggregatedFieldError(name, available)

        values: list[float] = []
        weights: list[float] = []
        nested_items: list[ModelResult] = []
        nested_lists: list[ListResult] = []

        for item in self._items:
            if name in item._fields:
                field = item._fields[name]
                if isinstance(field, FieldResult):
                    values.append(field.value)
                    weights.append(field.weight)
                elif isinstance(field, ModelResult):
                    nested_items.append(field)
                elif isinstance(field, ListResult):
                    # Nested list - collect for NestedListAggregatedResult
                    nested_lists.append(field)

        # If we have nested lists, return NestedListAggregatedResult
        if nested_lists:
            element_type = nested_lists[0]._element_type if nested_lists else None
            return NestedListAggregatedResult(nested_lists, element_type)

        if nested_items:
            return AggregatedModelResult(_items=nested_items)
        return AggregatedFieldResult(_values=values, _weights=weights)

    def __repr__(self) -> str:
        """Return a string representation with available fields."""
        if not self._items:
            return "AggregatedFieldResultCollection()"
        # Get available fields from first item
        available_fields = list(self._items[0]._fields.keys())
        fields_str = ", ".join(f"{name}=..." for name in available_fields)
        return f"AggregatedFieldResultCollection({fields_str})"


@dataclass
class ListResult(StatsMixin):
    """
    Result for a list[BaseModel] field.

    Provides two access modes:
    - By index: items[0] -> ModelResult
    - Aggregated (required): items.aggregated_fields.name ->
        AggregatedFieldResult
    """

    _items: list[ModelResult]
    weight: float = 1.0
    _element_type: type | None = None

    def _get_values(self) -> list[float]:
        """Get all values for statistical computation."""
        return self._collect_all_values()

    def _get_values_and_weights(
        self,
    ) -> tuple[list[float], list[float]]:
        """Get all values and weights for weighted statistical computation."""
        return self._collect_all_values_and_weights()

    def __getitem__(self, index: int) -> ModelResult:
        """
        Get fill rate result for a specific item by index.

        Args:
            index: The item index.

        Returns:
            ModelResult for the item.

        Raises:
            IndexError: If index is out of range.
        """
        return self._items[index]

    def __len__(self) -> int:
        """Get the number of items."""
        return len(self._items)

    def __iter__(self) -> t.Iterator[ModelResult]:
        """Iterate over all items."""
        return iter(self._items)

    @property
    def aggregated_fields(self) -> AggregatedFieldResultCollection:
        """
        Get aggregated fields collection for accessing fields across all items.

        Returns:
            AggregatedFieldResultCollection instance.
        """
        return AggregatedFieldResultCollection(self._items)

    @property
    def value(self) -> float:
        """
        Get the mean fill rate across all items.

        Returns:
            Mean fill rate.
        """
        return self.mean()

    def _collect_all_values(self) -> list[float]:
        """
        Collect all fill rate values from all items (recursively).

        Returns:
            List of all fill rate values.
        """
        values: list[float] = []
        for item in self._items:
            values.extend(item._collect_all_values())
        return values

    def _collect_all_values_and_weights(self) -> tuple[list[float], list[float]]:
        """
        Collect all fill rate values and weights from all items (recursively).

        Returns:
            Tuple of (values, weights) lists.
        """
        values: list[float] = []
        weights: list[float] = []
        for item in self._items:
            v, w = item._collect_all_values_and_weights()
            values.extend(v)
            weights.extend(w)
        return values, weights

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"ListResult(items={len(self._items)}, weight={self.weight})"


class ModelResultCollection:
    """
    Collection of ModelResult for aggregated statistics.

    Created by adding ModelResults together: result1 + result2
    Provides methods to aggregate results and export to pandas DataFrame.
    """

    def __init__(self, results: list[ModelResult]) -> None:
        """
        Initialize a ModelResultCollection.

        Args:
            results: List of ModelResult instances (must be from same model type).
        """
        self._results = results
        self._model_type: type | None = None

    def __add__(
        self, other: ModelResult | "ModelResultCollection"
    ) -> "ModelResultCollection":
        """
        Add a ModelResult or merge another collection.

        Args:
            other: ModelResult or ModelResultCollection to add.

        Returns:
            New ModelResultCollection with combined results.

        Raises:
            IncompatibleModelResultError: If results come from different model types.
        """
        from cobjectric.exceptions import IncompatibleModelResultError  # noqa: PLC0415

        if isinstance(other, ModelResult):
            # Validate compatibility
            if self._results:
                self_fields = set(self._results[0]._fields.keys())
                other_fields = set(other._fields.keys())
                if self_fields != other_fields:
                    raise IncompatibleModelResultError(
                        type(self._results[0]), type(other)
                    )
            return ModelResultCollection(self._results + [other])
        elif isinstance(other, ModelResultCollection):
            # Merge two collections
            if self._results and other._results:
                self_fields = set(self._results[0]._fields.keys())
                other_fields = set(other._results[0]._fields.keys())
                if self_fields != other_fields:
                    raise IncompatibleModelResultError(
                        type(self._results[0]), type(other._results[0])
                    )
            return ModelResultCollection(self._results + other._results)
        else:
            raise TypeError(
                f"Cannot add {type(other)} to ModelResultCollection. "
                "Only ModelResult and ModelResultCollection are supported."
            )

    def to_dataframe(self) -> t.Any:
        """
        Export to pandas DataFrame (requires cobjectric[pandas]).

        Each row represents one ModelResult, columns are field paths.

        Returns:
            pandas.DataFrame with one row per ModelResult.

        Raises:
            ImportError: If pandas is not installed.
        """
        from cobjectric.pandas_export import (  # noqa: PLC0415
            _flatten_to_dict,
            _get_pandas,
        )

        pd = _get_pandas()

        # Flatten each result
        rows: list[dict[str, float]] = []
        for result in self._results:
            flattened = _flatten_to_dict(result)
            rows.append(flattened)

        # Create DataFrame
        return pd.DataFrame(rows)

    def mean(self) -> dict[str, float]:
        """
        Calculate mean value for each field across all results.

        Returns:
            Dictionary mapping field paths to mean values.
        """
        if not self._results:
            return {}

        # Get all field paths from first result
        from cobjectric.pandas_export import _flatten_to_dict  # noqa: PLC0415

        first_flattened = _flatten_to_dict(self._results[0])
        field_paths = list(first_flattened.keys())

        # Calculate mean for each field
        means: dict[str, float] = {}
        for path in field_paths:
            values = []
            for result in self._results:
                flattened = _flatten_to_dict(result)
                if path in flattened:
                    values.append(flattened[path])
            if values:
                means[path] = sum(values) / len(values)

        return means

    def std(self) -> dict[str, float]:
        """
        Calculate standard deviation for each field across all results.

        Returns:
            Dictionary mapping field paths to standard deviation values.
        """
        if not self._results:
            return {}

        from cobjectric.pandas_export import _flatten_to_dict  # noqa: PLC0415

        first_flattened = _flatten_to_dict(self._results[0])
        field_paths = list(first_flattened.keys())

        stds: dict[str, float] = {}
        for path in field_paths:
            values = []
            for result in self._results:
                flattened = _flatten_to_dict(result)
                if path in flattened:
                    values.append(flattened[path])
            if len(values) <= 1:
                stds[path] = 0.0
            else:
                mean_val = sum(values) / len(values)
                variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                stds[path] = variance**0.5

        return stds

    def var(self) -> dict[str, float]:
        """
        Calculate variance for each field across all results.

        Returns:
            Dictionary mapping field paths to variance values.
        """
        if not self._results:
            return {}

        from cobjectric.pandas_export import _flatten_to_dict  # noqa: PLC0415

        first_flattened = _flatten_to_dict(self._results[0])
        field_paths = list(first_flattened.keys())

        vars_dict: dict[str, float] = {}
        for path in field_paths:
            values = []
            for result in self._results:
                flattened = _flatten_to_dict(result)
                if path in flattened:
                    values.append(flattened[path])
            if len(values) <= 1:
                vars_dict[path] = 0.0
            else:
                mean_val = sum(values) / len(values)
                vars_dict[path] = sum((x - mean_val) ** 2 for x in values) / len(values)

        return vars_dict

    def min(self) -> dict[str, float]:
        """
        Get minimum value for each field across all results.

        Returns:
            Dictionary mapping field paths to minimum values.
        """
        if not self._results:
            return {}

        from cobjectric.pandas_export import _flatten_to_dict  # noqa: PLC0415

        first_flattened = _flatten_to_dict(self._results[0])
        field_paths = list(first_flattened.keys())

        mins: dict[str, float] = {}
        for path in field_paths:
            values = []
            for result in self._results:
                flattened = _flatten_to_dict(result)
                if path in flattened:
                    values.append(flattened[path])
            if values:
                mins[path] = min(values)

        return mins

    def max(self) -> dict[str, float]:
        """
        Get maximum value for each field across all results.

        Returns:
            Dictionary mapping field paths to maximum values.
        """
        if not self._results:
            return {}

        from cobjectric.pandas_export import _flatten_to_dict  # noqa: PLC0415

        first_flattened = _flatten_to_dict(self._results[0])
        field_paths = list(first_flattened.keys())

        maxs: dict[str, float] = {}
        for path in field_paths:
            values = []
            for result in self._results:
                flattened = _flatten_to_dict(result)
                if path in flattened:
                    values.append(flattened[path])
            if values:
                maxs[path] = max(values)

        return maxs

    def quantile(self, q: float) -> dict[str, float]:
        """
        Calculate quantile for each field across all results.

        Args:
            q: The quantile to compute (float between 0.0 and 1.0).

        Returns:
            Dictionary mapping field paths to quantile values.

        Raises:
            ValueError: If q is not in [0, 1].
        """
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantile q must be between 0.0 and 1.0, got {q}")

        if not self._results:
            return {}

        from cobjectric.pandas_export import _flatten_to_dict  # noqa: PLC0415

        first_flattened = _flatten_to_dict(self._results[0])
        field_paths = list(first_flattened.keys())

        quantiles: dict[str, float] = {}
        for path in field_paths:
            values = []
            for result in self._results:
                flattened = _flatten_to_dict(result)
                if path in flattened:
                    values.append(flattened[path])

            if not values:
                continue

            sorted_values = sorted(values)
            n = len(sorted_values)

            if q == 0.0:
                quantiles[path] = sorted_values[0]
            elif q == 1.0:
                quantiles[path] = sorted_values[-1]
            else:
                index = q * (n - 1)
                lower_index = int(index)
                upper_index = min(lower_index + 1, n - 1)
                weight = index - lower_index
                quantiles[path] = (
                    sorted_values[lower_index] * (1 - weight)
                    + sorted_values[upper_index] * weight
                )

        return quantiles

    def __len__(self) -> int:
        """Get the number of results in the collection."""
        return len(self._results)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"ModelResultCollection(results={len(self._results)})"
