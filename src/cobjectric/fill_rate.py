from __future__ import annotations

import typing as t
from dataclasses import dataclass

from cobjectric.exceptions import InvalidWeightError

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
        self, fields: dict[str, FillRateFieldResult | FillRateModelResult]
    ) -> None:
        """
        Initialize a FillRateFieldCollection.

        Args:
            fields: Dictionary mapping field names to FillRateFieldResult or
                FillRateModelResult instances.
        """
        self._fields = fields

    def __getattr__(self, name: str) -> FillRateFieldResult | FillRateModelResult:
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
    ) -> t.Iterator[FillRateFieldResult | FillRateModelResult]:
        """Iterate over all fill rate results."""
        return iter(self._fields.values())

    def __getitem__(self, path: str) -> FillRateFieldResult | FillRateModelResult:
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
    ) -> FillRateFieldResult | FillRateModelResult:
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

        current: FillRateFieldResult | FillRateModelResult | None = None
        current_fields = self._fields

        for i, segment in enumerate(segments):
            if segment.startswith("[") and segment.endswith("]"):
                # List index - not yet supported for fill rate results
                raise KeyError(
                    f"List index access not yet supported "
                    f"in fill rate results: {segment}"
                )

            if segment not in current_fields:
                raise KeyError(f"Field '{segment}' not found in path")

            current = current_fields[segment]

            if i < len(segments) - 1:
                # More segments to go
                next_segment = segments[i + 1]
                # Check if next segment is a list index
                if next_segment.startswith("[") and next_segment.endswith("]"):
                    # List index - not yet supported for fill rate results
                    raise KeyError(
                        f"List index access not yet supported "
                        f"in fill rate results: {next_segment}"
                    )

                # Current must be a FillRateModelResult
                if not isinstance(current, FillRateModelResult):
                    raise KeyError(
                        f"Cannot access '{next_segment}' "
                        f"on non-model field '{segment}'"
                    )
                current_fields = current._fields

        assert current is not None, "Invalid path"
        return current

    def __repr__(self) -> str:
        """Return a string representation of the FillRateFieldCollection."""
        fields_repr = ", ".join(
            f"{name}={field!r}" for name, field in self._fields.items()
        )
        return f"FillRateFieldCollection({fields_repr})"


@dataclass
class FillRateModelResult:
    """
    Result of fill rate computation for a model instance.

    Attributes:
        _fields: Dictionary mapping field names to FillRateFieldResult or
            FillRateModelResult instances (for nested models).
    """

    _fields: dict[str, FillRateFieldResult | FillRateModelResult]

    @property
    def fields(self) -> FillRateFieldCollection:
        """
        Get the FillRateFieldCollection for this result.

        Returns:
            The FillRateFieldCollection containing all fill rate results.
        """
        return FillRateFieldCollection(self._fields)

    def __getitem__(self, path: str) -> FillRateFieldResult | FillRateModelResult:
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
