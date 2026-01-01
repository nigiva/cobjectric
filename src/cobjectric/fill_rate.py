from __future__ import annotations

import typing as t
from dataclasses import dataclass

from cobjectric.exceptions import InvalidAggregatedFieldError, InvalidWeightError

FillRateFunc = t.Callable[[t.Any], float]


@dataclass
class FillRateFuncInfo:
    """
    Stores fill_rate_func info attached to a method.

    Attributes:
        field_patterns: Tuple of field names or glob patterns to match.
        func: The fill_rate_func to apply.
        weight: Weight for fill rate computation (default: 1.0, must be >= 0.0).
    """

    field_patterns: tuple[str, ...]
    func: FillRateFunc
    weight: float = 1.0


def fill_rate_func(
    *field_patterns: str,
    weight: float = 1.0,
) -> t.Callable[[FillRateFunc], FillRateFunc]:
    """
    Decorator to define a fill_rate_func for one or more fields.

    Args:
        *field_patterns: Field names or glob patterns (e.g., "name", "email", "name_*")
        weight: Weight for fill rate computation (default: 1.0, must be >= 0.0).

    Returns:
        Decorated function

    Raises:
        InvalidWeightError: If weight is negative (< 0.0).

    Example:
        ```python
        class Person(BaseModel):
            name: str
            email: str

            @fill_rate_func("name", "email", weight=2.0)
            def fill_rate_name_email(x: t.Any) -> float:
                return len(x) / 100 if x is not MissingValue else 0.0
        ```
    """
    if weight < 0.0:
        raise InvalidWeightError(weight, "decorator")

    def decorator(func: FillRateFunc) -> FillRateFunc:
        if not hasattr(func, "_fill_rate_funcs"):
            func._fill_rate_funcs = []  # type: ignore[attr-defined]
        func._fill_rate_funcs.append(  # type: ignore[attr-defined]
            FillRateFuncInfo(field_patterns, func, weight)
        )
        return func

    return decorator


FillRateAccuracyFunc = t.Callable[[t.Any, t.Any], float]


@dataclass
class FillRateAccuracyFuncInfo:
    """
    Stores fill_rate_accuracy_func info attached to a method.

    Attributes:
        field_patterns: Tuple of field names or glob patterns to match.
        func: The fill_rate_accuracy_func to apply.
        weight: Weight for fill rate accuracy computation
            (default: 1.0, must be >= 0.0).
    """

    field_patterns: tuple[str, ...]
    func: FillRateAccuracyFunc
    weight: float = 1.0


def fill_rate_accuracy_func(
    *field_patterns: str,
    weight: float = 1.0,
) -> t.Callable[[FillRateAccuracyFunc], FillRateAccuracyFunc]:
    """
    Decorator to define a fill_rate_accuracy_func for one or more fields.

    Args:
        *field_patterns: Field names or glob patterns (e.g., "name", "email", "name_*")
        weight: Weight for fill rate accuracy computation
            (default: 1.0, must be >= 0.0).

    Returns:
        Decorated function

    Raises:
        InvalidWeightError: If weight is negative (< 0.0).

    Example:
        ```python
        class Person(BaseModel):
            name: str
            email: str

            @fill_rate_accuracy_func("name", "email", weight=2.0)
            def accuracy_name_email(got: t.Any, expected: t.Any) -> float:
                return (
                    1.0
                    if (got is not MissingValue) == (expected is not MissingValue)
                    else 0.0
                )
        ```
    """
    if weight < 0.0:
        raise InvalidWeightError(weight, "decorator", "fill_rate_accuracy")

    def decorator(func: FillRateAccuracyFunc) -> FillRateAccuracyFunc:
        if not hasattr(func, "_fill_rate_accuracy_funcs"):
            func._fill_rate_accuracy_funcs = []  # type: ignore[attr-defined]
        func._fill_rate_accuracy_funcs.append(  # type: ignore[attr-defined]
            FillRateAccuracyFuncInfo(field_patterns, func, weight)
        )
        return func

    return decorator


SimilarityFunc = t.Callable[[t.Any, t.Any], float]


@dataclass
class SimilarityFuncInfo:
    """
    Stores similarity_func info attached to a method.

    Attributes:
        field_patterns: Tuple of field names or glob patterns to match.
        func: The similarity_func to apply.
        weight: Weight for similarity computation (default: 1.0, must be >= 0.0).
    """

    field_patterns: tuple[str, ...]
    func: SimilarityFunc
    weight: float = 1.0


def similarity_func(
    *field_patterns: str,
    weight: float = 1.0,
) -> t.Callable[[SimilarityFunc], SimilarityFunc]:
    """
    Decorator to define a similarity_func for one or more fields.

    Args:
        *field_patterns: Field names or glob patterns (e.g., "name", "email", "name_*")
        weight: Weight for similarity computation (default: 1.0, must be >= 0.0).

    Returns:
        Decorated function

    Raises:
        InvalidWeightError: If weight is negative (< 0.0).

    Example:
        ```python
        class Person(BaseModel):
            name: str
            email: str

            @similarity_func("name", "email", weight=2.0)
            def similarity_name_email(x: t.Any, y: t.Any) -> float:
                return 1.0 if x == y else 0.0
        ```
    """
    if weight < 0.0:
        raise InvalidWeightError(weight, "decorator", "similarity")

    def decorator(func: SimilarityFunc) -> SimilarityFunc:
        if not hasattr(func, "_similarity_funcs"):
            func._similarity_funcs = []  # type: ignore[attr-defined]
        func._similarity_funcs.append(  # type: ignore[attr-defined]
            SimilarityFuncInfo(field_patterns, func, weight)
        )
        return func

    return decorator


@dataclass
class FillRateFieldResult:
    """
    Result of fill rate computation for a single field.

    Attributes:
        value: The fill rate value (float between 0.0 and 1.0).
        weight: Weight for this field in weighted mean calculation (default: 1.0).
    """

    value: float
    weight: float = 1.0

    def __repr__(self) -> str:
        """Return a string representation of the FillRateFieldResult."""
        return f"FillRateFieldResult(value={self.value}, weight={self.weight})"


class FillRateFieldCollection:
    """
    Collection of fill rate results for a model instance.

    Provides attribute-based access to fill rate results.
    """

    def __init__(
        self,
        fields: dict[
            str,
            FillRateFieldResult | FillRateModelResult | FillRateListResult,
        ],
    ) -> None:
        """
        Initialize a FillRateFieldCollection.

        Args:
            fields: Dictionary mapping field names to FillRateFieldResult,
                FillRateModelResult, or FillRateListResult instances.
        """
        self._fields = fields

    def __getattr__(
        self, name: str
    ) -> FillRateFieldResult | FillRateModelResult | FillRateListResult:
        """
        Get a fill rate result by field name.

        Args:
            name: The name of the field.

        Returns:
            The FillRateFieldResult or FillRateModelResult instance.

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
    ) -> t.Iterator[FillRateFieldResult | FillRateModelResult | FillRateListResult]:
        """Iterate over all fill rate results."""
        return iter(self._fields.values())

    def __getitem__(
        self, path: str
    ) -> FillRateFieldResult | FillRateModelResult | FillRateListResult:
        """
        Get a fill rate result by path.

        Args:
            path: Path to the field (e.g., "name", "address.city", "items[0].name").

        Returns:
            The FillRateFieldResult or FillRateModelResult instance.

        Raises:
            KeyError: If the path is invalid.
        """
        segments = self._parse_path(path)
        return self._resolve_path(segments)

    def _parse_path(self, path: str) -> list[str]:
        """
        Parse a path string into segments.

        Args:
            path: Path string (e.g., "address.city", "items[0].name").

        Returns:
            List of path segments.
        """
        segments: list[str] = []
        current = ""
        i = 0
        while i < len(path):
            if path[i] == ".":
                if current:
                    segments.append(current)
                    current = ""
            elif path[i] == "[":
                if current:
                    segments.append(current)
                    current = ""
                # Find closing bracket
                j = i + 1
                while j < len(path) and path[j] != "]":
                    j += 1
                if j >= len(path):
                    raise KeyError(f"Invalid path: {path}")
                index_str = path[i + 1 : j]
                try:
                    index = int(index_str)
                    segments.append(f"[{index}]")
                except ValueError as e:
                    raise KeyError(f"Invalid path: {path}") from e
                i = j
            else:
                current += path[i]
            i += 1
        if current:
            segments.append(current)
        return segments

    def _resolve_path(
        self, segments: list[str]
    ) -> FillRateFieldResult | FillRateModelResult | FillRateListResult:
        """
        Resolve a path from segments.

        Args:
            segments: List of path segments.

        Returns:
            The FillRateFieldResult or FillRateModelResult instance.

        Raises:
            KeyError: If the path is invalid.
        """
        if not segments:
            raise KeyError("Empty path")

        current: (
            FillRateFieldResult | FillRateModelResult | FillRateListResult | None
        ) = None
        current_fields = self._fields

        for i, segment in enumerate(segments):
            if segment.startswith("[") and segment.endswith("]"):
                # List index access
                index_str = segment[1:-1]
                try:
                    index = int(index_str)
                except ValueError as e:
                    raise KeyError(f"Invalid list index: {index_str}") from e

                if isinstance(current, FillRateListResult):
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
                    if isinstance(current, FillRateListResult):
                        # current_fields stays as is, updated when accessing index
                        continue
                    raise KeyError(f"Cannot use index on non-list field '{segment}'")

                # Current must be a FillRateModelResult or FillRateListResult
                if isinstance(current, FillRateModelResult):
                    current_fields = current._fields
                elif isinstance(current, FillRateListResult):
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
        """Return a string representation of the FillRateFieldCollection."""
        fields_repr = ", ".join(f"{name}=..." for name in self._fields.keys())
        return f"FillRateFieldCollection({fields_repr})"


@dataclass
class FillRateModelResult:
    """
    Result of fill rate computation for a model instance.

    Attributes:
        _fields: Dictionary mapping field names to FillRateFieldResult,
            FillRateModelResult, or FillRateListResult instances.
    """

    _fields: dict[str, FillRateFieldResult | FillRateModelResult | FillRateListResult]

    @property
    def fields(self) -> FillRateFieldCollection:
        """
        Get the FillRateFieldCollection for this result.

        Returns:
            The FillRateFieldCollection containing all fill rate results.
        """
        return FillRateFieldCollection(self._fields)

    def __getitem__(
        self, path: str
    ) -> FillRateFieldResult | FillRateModelResult | FillRateListResult:
        """
        Get a fill rate result by path.

        Args:
            path: Path to the field (e.g., "name", "address.city", "items[0].name").

        Returns:
            The FillRateFieldResult or FillRateModelResult instance.

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
            if isinstance(field_result, FillRateModelResult):
                values.extend(field_result._collect_all_values())
            elif isinstance(field_result, FillRateListResult):
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
            if isinstance(field_result, FillRateModelResult):
                nested_values, nested_weights = (
                    field_result._collect_all_values_and_weights()
                )
                values.extend(nested_values)
                weights.extend(nested_weights)
            elif isinstance(field_result, FillRateListResult):
                for item in field_result._items:
                    item_values, item_weights = item._collect_all_values_and_weights()
                    values.extend(item_values)
                    weights.extend(item_weights)
            else:
                values.append(field_result.value)
                weights.append(field_result.weight)
        return values, weights

    def mean(self) -> float:
        """
        Calculate the weighted mean fill rate across all fields.

        Returns:
            The weighted mean fill rate (float between 0.0 and 1.0).
            Formula: sum(value * weight) / sum(weight)
        """
        values, weights = self._collect_all_values_and_weights()
        if not values:
            return 0.0
        total_weight = sum(weights)
        if total_weight == 0.0:
            return 0.0
        return sum(v * w for v, w in zip(values, weights, strict=True)) / total_weight

    def max(self) -> float:
        """
        Get the maximum fill rate across all fields.

        Returns:
            The maximum fill rate (float between 0.0 and 1.0).
        """
        values = self._collect_all_values()
        if not values:
            return 0.0
        return max(values)

    def min(self) -> float:
        """
        Get the minimum fill rate across all fields.

        Returns:
            The minimum fill rate (float between 0.0 and 1.0).
        """
        values = self._collect_all_values()
        if not values:
            return 0.0
        return min(values)

    def std(self) -> float:
        """
        Calculate the standard deviation of fill rates across all fields.

        Returns:
            The standard deviation (float >= 0.0).
        """
        values = self._collect_all_values()
        if not values:
            return 0.0
        if len(values) == 1:
            return 0.0
        mean_val = self.mean()
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance**0.5

    def var(self) -> float:
        """
        Calculate the variance of fill rates across all fields.

        Returns:
            The variance (float >= 0.0).
        """
        values = self._collect_all_values()
        if not values:
            return 0.0
        if len(values) == 1:
            return 0.0
        mean_val = self.mean()
        return sum((x - mean_val) ** 2 for x in values) / len(values)

    def quantile(self, q: float) -> float:
        """
        Calculate the quantile of fill rates across all fields.

        Args:
            q: The quantile to compute (float between 0.0 and 1.0).

        Returns:
            The quantile value (float between 0.0 and 1.0).

        Raises:
            ValueError: If q is not in [0, 1].
        """
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantile q must be between 0.0 and 1.0, got {q}")

        values = self._collect_all_values()
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if q == 0.0:
            return sorted_values[0]
        if q == 1.0:
            return sorted_values[-1]

        index = q * (n - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, n - 1)
        weight = index - lower_index

        return (
            sorted_values[lower_index] * (1 - weight)
            + sorted_values[upper_index] * weight
        )

    def __repr__(self) -> str:
        """Return a string representation of the FillRateModelResult."""
        fields_repr = ", ".join(
            f"{name}={field!r}" for name, field in self._fields.items()
        )
        return f"FillRateModelResult({fields_repr})"


@dataclass
class FillRateAggregatedFieldResult:
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

    def mean(self) -> float:
        """
        Calculate weighted mean of fill rates.

        Returns:
            Weighted mean fill rate.
        """
        if not self._values:
            return 0.0
        total_weight = sum(self._weights)
        if total_weight == 0.0:
            return 0.0
        return (
            sum(v * w for v, w in zip(self._values, self._weights, strict=True))
            / total_weight
        )

    def max(self) -> float:
        """
        Get maximum fill rate value.

        Returns:
            Maximum fill rate.
        """
        return max(self._values) if self._values else 0.0

    def min(self) -> float:
        """
        Get minimum fill rate value.

        Returns:
            Minimum fill rate.
        """
        return min(self._values) if self._values else 0.0

    def std(self) -> float:
        """
        Calculate standard deviation of fill rates.

        Returns:
            Standard deviation.
        """
        if len(self._values) <= 1:
            return 0.0
        m = self.mean()
        return (sum((x - m) ** 2 for x in self._values) / len(self._values)) ** 0.5

    def var(self) -> float:
        """
        Calculate variance of fill rates.

        Returns:
            Variance.
        """
        if len(self._values) <= 1:
            return 0.0
        m = self.mean()
        return sum((x - m) ** 2 for x in self._values) / len(self._values)

    def quantile(self, q: float) -> float:
        """
        Calculate quantile of fill rates.

        Args:
            q: The quantile to compute (float between 0.0 and 1.0).

        Returns:
            The quantile value.

        Raises:
            ValueError: If q is not in [0, 1].
        """
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantile q must be between 0.0 and 1.0, got {q}")

        if not self._values:
            return 0.0

        sorted_values = sorted(self._values)
        n = len(sorted_values)

        if q == 0.0:
            return sorted_values[0]
        if q == 1.0:
            return sorted_values[-1]

        index = q * (n - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, n - 1)
        weight = index - lower_index

        return (
            sorted_values[lower_index] * (1 - weight)
            + sorted_values[upper_index] * weight
        )

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"FillRateAggregatedFieldResult(values={self._values})"


@dataclass
class FillRateAggregatedModelResult:
    """
    Aggregated result for a nested model field across list items.

    Allows chained access: items.aggregated_fields.address.city

    Note:
        For nested lists (list[list[BaseModel]]), accessing a nested list field
        returns the mean fill rate of each list, not individual field access.
        For example, if items have a tags field of type list[Tag], accessing
        items.aggregated_fields.tags returns a FillRateAggregatedFieldResult
        with the mean fill rate of each tags list, not access to individual
        tag fields.
    """

    _items: list[FillRateModelResult]

    def __getattr__(
        self, name: str
    ) -> (
        FillRateAggregatedFieldResult
        | FillRateAggregatedModelResult
        | FillRateNestedListAggregatedResult
    ):
        """
        Get aggregated result for a field name across items.

        Args:
            name: The field name.

        Returns:
            FillRateAggregatedFieldResult, FillRateAggregatedModelResult,
            or FillRateNestedListAggregatedResult.

        Raises:
            InvalidAggregatedFieldError: If the field doesn't exist in the model.
        """
        # Validate field exists
        if self._items and name not in self._items[0]._fields:
            available = list(self._items[0]._fields.keys())
            raise InvalidAggregatedFieldError(name, available)

        values: list[float] = []
        weights: list[float] = []
        nested_items: list[FillRateModelResult] = []
        nested_lists: list[FillRateListResult] = []

        for item in self._items:
            if name in item._fields:
                field = item._fields[name]
                if isinstance(field, FillRateFieldResult):
                    values.append(field.value)
                    weights.append(field.weight)
                elif isinstance(field, FillRateModelResult):
                    nested_items.append(field)
                elif isinstance(field, FillRateListResult):
                    # Nested list - collect for FillRateNestedListAggregatedResult
                    nested_lists.append(field)

        # If we have nested lists, return FillRateNestedListAggregatedResult
        if nested_lists:
            element_type = nested_lists[0]._element_type if nested_lists else None
            return FillRateNestedListAggregatedResult(nested_lists, element_type)

        if nested_items:
            return FillRateAggregatedModelResult(_items=nested_items)
        return FillRateAggregatedFieldResult(_values=values, _weights=weights)

    def mean(self) -> float:
        """
        Calculate mean fill rate across all items.

        Returns:
            Mean fill rate.
        """
        if not self._items:
            return 0.0
        return sum(item.mean() for item in self._items) / len(self._items)

    def max(self) -> float:
        """
        Get maximum fill rate across all items.

        Returns:
            Maximum fill rate.
        """
        if not self._items:
            return 0.0
        return max(item.mean() for item in self._items)

    def min(self) -> float:
        """
        Get minimum fill rate across all items.

        Returns:
            Minimum fill rate.
        """
        if not self._items:
            return 0.0
        return min(item.mean() for item in self._items)

    def std(self) -> float:
        """
        Calculate standard deviation of fill rates across items.

        Returns:
            Standard deviation.
        """
        if len(self._items) <= 1:
            return 0.0
        means = [item.mean() for item in self._items]
        m = self.mean()
        return (sum((x - m) ** 2 for x in means) / len(means)) ** 0.5

    def var(self) -> float:
        """
        Calculate variance of fill rates across items.

        Returns:
            Variance.
        """
        if len(self._items) <= 1:
            return 0.0
        means = [item.mean() for item in self._items]
        m = self.mean()
        return sum((x - m) ** 2 for x in means) / len(means)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"FillRateAggregatedModelResult(items={len(self._items)})"


class FillRateNestedListAggregatedResult:
    """
    Aggregated result for nested lists in fill rate results.

    When accessing a list field through aggregated_fields, this allows
    chaining to further aggregate nested lists and access their fields.
    """

    def __init__(
        self, lists: list[FillRateListResult], element_type: type | None = None
    ) -> None:
        """
        Initialize FillRateNestedListAggregatedResult.

        Args:
            lists: List of FillRateListResult instances.
            element_type: The element type of the list (optional).
        """
        self._lists = lists
        self._element_type = element_type

    @property
    def aggregated_fields(self) -> FillRateAggregatedFieldCollection:
        """
        Get aggregated fields across all nested lists.

        Flattens the nested lists and returns a collection for accessing
        fields across all items.

        Returns:
            FillRateAggregatedFieldCollection instance.
        """
        # Flatten all items from all lists
        all_items: list[FillRateModelResult] = []
        for list_result in self._lists:
            all_items.extend(list_result._items)
        return FillRateAggregatedFieldCollection(all_items)

    @property
    def values(self) -> list[float]:
        """
        Get fill rate values for nested lists.

        By default, returns the mean fill rate of each list (one value per list).
        This is the hierarchical view. To get flattened values, use
        FillRateNestedListAggregatedResult.aggregated_fields to access individual
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
        return f"FillRateNestedListAggregatedResult(lists={len(self._lists)})"


class FillRateAggregatedFieldCollection:
    """
    Collection for aggregated field access on a list of fill rate results.

    Provides access to aggregated field results across all items
    in the list.

    Note:
        For nested lists (list[list[BaseModel]]), accessing a nested list
        field returns the mean fill rate of each list, not individual field
        access. Use indexed access to access nested list items individually.
    """

    def __init__(self, items: list[FillRateModelResult]) -> None:
        """
        Initialize the collection.

        Args:
            items: List of FillRateModelResult instances.
        """
        self._items = items

    def _get_available_fields(self) -> list[str]:
        """Get list of available field names from first item."""
        if self._items:
            return list(self._items[0]._fields.keys())
        return []

    def __getattr__(
        self, name: str
    ) -> (
        FillRateAggregatedFieldResult
        | FillRateAggregatedModelResult
        | FillRateNestedListAggregatedResult
    ):
        """
        Get aggregated result for a field name across all items.

        Args:
            name: The field name.

        Returns:
            FillRateAggregatedFieldResult, FillRateAggregatedModelResult,
            or FillRateNestedListAggregatedResult.

        Raises:
            InvalidAggregatedFieldError: If the field doesn't exist in the model.
        """
        # Validate field exists
        if self._items and name not in self._items[0]._fields:
            available = self._get_available_fields()
            raise InvalidAggregatedFieldError(name, available)

        values: list[float] = []
        weights: list[float] = []
        nested_items: list[FillRateModelResult] = []
        nested_lists: list[FillRateListResult] = []

        for item in self._items:
            if name in item._fields:
                field = item._fields[name]
                if isinstance(field, FillRateFieldResult):
                    values.append(field.value)
                    weights.append(field.weight)
                elif isinstance(field, FillRateModelResult):
                    nested_items.append(field)
                elif isinstance(field, FillRateListResult):
                    # Nested list - collect for FillRateNestedListAggregatedResult
                    nested_lists.append(field)

        # If we have nested lists, return FillRateNestedListAggregatedResult
        if nested_lists:
            element_type = nested_lists[0]._element_type if nested_lists else None
            return FillRateNestedListAggregatedResult(nested_lists, element_type)

        if nested_items:
            return FillRateAggregatedModelResult(_items=nested_items)
        return FillRateAggregatedFieldResult(_values=values, _weights=weights)

    def __repr__(self) -> str:
        """Return a string representation with available fields."""
        if not self._items:
            return "FillRateAggregatedFieldCollection()"
        # Get available fields from first item
        available_fields = list(self._items[0]._fields.keys())
        fields_str = ", ".join(f"{name}=..." for name in available_fields)
        return f"FillRateAggregatedFieldCollection({fields_str})"


@dataclass
class FillRateListResult:
    """
    Result for a list[BaseModel] field.

    Provides two access modes:
    - By index: items[0] -> FillRateModelResult
    - Aggregated (required): items.aggregated_fields.name ->
        FillRateAggregatedFieldResult
    """

    _items: list[FillRateModelResult]
    weight: float = 1.0
    _element_type: type | None = None

    def __getitem__(self, index: int) -> FillRateModelResult:
        """
        Get fill rate result for a specific item by index.

        Args:
            index: The item index.

        Returns:
            FillRateModelResult for the item.

        Raises:
            IndexError: If index is out of range.
        """
        return self._items[index]

    def __len__(self) -> int:
        """Get the number of items."""
        return len(self._items)

    def __iter__(self) -> t.Iterator[FillRateModelResult]:
        """Iterate over all items."""
        return iter(self._items)

    @property
    def aggregated_fields(self) -> FillRateAggregatedFieldCollection:
        """
        Get aggregated fields collection for accessing fields across all items.

        Returns:
            FillRateAggregatedFieldCollection instance.
        """
        return FillRateAggregatedFieldCollection(self._items)

    @property
    def value(self) -> float:
        """
        Get the mean fill rate across all items.

        Returns:
            Mean fill rate.
        """
        return self.mean()

    def mean(self) -> float:
        """
        Calculate weighted mean fill rate across all items.

        Returns:
            Weighted mean fill rate.
        """
        if not self._items:
            return 0.0
        all_values: list[float] = []
        all_weights: list[float] = []
        for item in self._items:
            v, w = item._collect_all_values_and_weights()
            all_values.extend(v)
            all_weights.extend(w)
        if not all_values:
            return 0.0
        total_weight = sum(all_weights)
        if total_weight == 0.0:
            return 0.0
        return (
            sum(v * w for v, w in zip(all_values, all_weights, strict=True))
            / total_weight
        )

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
        return f"FillRateListResult(items={len(self._items)}, weight={self.weight})"
