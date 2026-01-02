class StatsMixin:
    """
    Mixin providing statistical methods for metric results.

    Subclasses must implement either:
    - `_get_values()` and `_get_values_and_weights()`, or
    - Override the statistical methods directly.
    """

    def _get_values(self) -> list[float]:
        """
        Get all values for statistical computation.

        Returns:
            List of float values.
        """
        raise NotImplementedError(
            "Subclass must implement _get_values() or override statistical methods"
        )

    def _get_values_and_weights(
        self,
    ) -> tuple[list[float], list[float]]:
        """
        Get all values and weights for weighted statistical computation.

        Returns:
            Tuple of (values, weights) lists.
        """
        raise NotImplementedError(
            "Subclass must implement _get_values_and_weights() "
            "or override statistical methods"
        )

    def mean(self) -> float:
        """
        Calculate the weighted mean.

        Returns:
            The weighted mean (float between 0.0 and 1.0).
            Formula: sum(value * weight) / sum(weight)
        """
        values, weights = self._get_values_and_weights()
        if not values:
            return 0.0
        total_weight = sum(weights)
        if total_weight == 0.0:
            return 0.0
        return sum(v * w for v, w in zip(values, weights, strict=True)) / total_weight

    def max(self) -> float:
        """
        Get the maximum value.

        Returns:
            The maximum value (float between 0.0 and 1.0).
        """
        values = self._get_values()
        if not values:
            return 0.0
        return max(values)

    def min(self) -> float:
        """
        Get the minimum value.

        Returns:
            The minimum value (float between 0.0 and 1.0).
        """
        values = self._get_values()
        if not values:
            return 0.0
        return min(values)

    def std(self) -> float:
        """
        Calculate the standard deviation.

        Returns:
            The standard deviation (float >= 0.0).
        """
        values = self._get_values()
        if not values:
            return 0.0
        if len(values) == 1:
            return 0.0
        mean_val = self.mean()
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance**0.5

    def var(self) -> float:
        """
        Calculate the variance.

        Returns:
            The variance (float >= 0.0).
        """
        values = self._get_values()
        if not values:
            return 0.0
        if len(values) == 1:
            return 0.0
        mean_val = self.mean()
        return sum((x - mean_val) ** 2 for x in values) / len(values)

    def quantile(self, q: float) -> float:
        """
        Calculate the quantile.

        Args:
            q: The quantile to compute (float between 0.0 and 1.0).

        Returns:
            The quantile value (float between 0.0 and 1.0).

        Raises:
            ValueError: If q is not in [0, 1].
        """
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantile q must be between 0.0 and 1.0, got {q}")

        values = self._get_values()
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
