from __future__ import annotations

import typing as t
from enum import Enum

from cobjectric.fill_rate import FillRateModelResult


class ListCompareStrategy(str, Enum):
    """
    Strategy for comparing list[BaseModel] items.

    Attributes:
        PAIRWISE: Compare items by their index (default).
        LEVENSHTEIN: Align items based on Levenshtein distance + similarity.
        OPTIMAL_ASSIGNMENT: Find optimal one-to-one mapping using Hungarian algorithm.
    """

    PAIRWISE = "pairwise"
    LEVENSHTEIN = "levenshtein"
    OPTIMAL_ASSIGNMENT = "optimal_assignment"


def align_pairwise(
    got_list: list[t.Any], expected_list: list[t.Any]
) -> list[tuple[int, int]]:
    """
    Align items pairwise by index.

    Args:
        got_list: List of BaseModel instances from the model being evaluated.
        expected_list: List of BaseModel instances from the expected model.

    Returns:
        List of tuples (got_index, expected_index) representing the alignment.
    """
    min_len = min(len(got_list), len(expected_list))
    return [(i, i) for i in range(min_len)]


def align_levenshtein(
    got_list: list[t.Any],
    expected_list: list[t.Any],
    compute_similarity_func: t.Callable[[t.Any, t.Any], FillRateModelResult],
) -> list[tuple[int, int]]:
    """
    Align items using Levenshtein-based alignment with similarity.

    Uses dynamic programming to find the alignment that maximizes total
    similarity while preserving relative order. Only pairs with positive
    similarity (> 0) are aligned. Items with zero similarity are skipped.

    Similar to Longest Common Subsequence (LCS) but weighted by similarity.

    Args:
        got_list: List of BaseModel instances from the model being evaluated.
        expected_list: List of BaseModel instances from the expected model.
        compute_similarity_func: Function to compute similarity between two
            BaseModel instances.

    Returns:
        List of tuples (got_index, expected_index) representing the alignment.
        Only includes pairs with positive similarity.
    """
    if not got_list or not expected_list:
        return []

    m, n = len(got_list), len(expected_list)

    # Build similarity matrix
    similarity_matrix: list[list[float]] = []
    for i in range(m):
        row: list[float] = []
        for j in range(n):
            similarity_result = compute_similarity_func(got_list[i], expected_list[j])
            similarity = similarity_result.mean()
            row.append(similarity)
        similarity_matrix.append(row)

    # DP table for maximum total similarity
    dp: list[list[float]] = [[0.0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table - only consider matches with similarity > 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sim = similarity_matrix[i - 1][j - 1]
            if sim > 0:
                match_score = dp[i - 1][j - 1] + sim
            else:
                match_score = float("-inf")

            delete_score = dp[i - 1][j]
            insert_score = dp[i][j - 1]

            dp[i][j] = max(match_score, delete_score, insert_score)

    # Backtrack to find alignment
    alignment: list[tuple[int, int]] = []
    i, j = m, n

    while i > 0 and j > 0:
        sim = similarity_matrix[i - 1][j - 1]
        current = dp[i][j]
        delete_score = dp[i - 1][j]
        insert_score = dp[i][j - 1]

        if sim > 0:
            match_score = dp[i - 1][j - 1] + sim
            if abs(current - match_score) < 1e-9:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue

        if abs(current - delete_score) < 1e-9:
            i -= 1
            continue

        if abs(current - insert_score) < 1e-9:
            j -= 1
            continue

        # Fallback: prefer delete over insert for consistency
        # no cover because it's a fallback and should never happen due to
        # the dp table is filled with the max of match_score, delete_score, insert_score
        if delete_score >= insert_score:  # pragma: no cover
            i -= 1
        else:  # pragma: no cover
            j -= 1

    alignment.reverse()
    return alignment


def align_optimal_assignment(
    got_list: list[t.Any],
    expected_list: list[t.Any],
    compute_similarity_func: t.Callable[[t.Any, t.Any], FillRateModelResult],
) -> list[tuple[int, int]]:
    """
    Align items using optimal assignment (Hungarian algorithm).

    Args:
        got_list: List of BaseModel instances from the model being evaluated.
        expected_list: List of BaseModel instances from the expected model.
        compute_similarity_func: Function to compute similarity between two
            BaseModel instances.

    Returns:
        List of tuples (got_index, expected_index) representing the alignment.
    """
    if not got_list or not expected_list:
        return []

    try:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "scipy is required for optimal_assignment strategy. "
            "Install it with: pip install scipy"
        ) from e

    m, n = len(got_list), len(expected_list)

    # Build similarity matrix
    similarity_matrix: list[list[float]] = []
    for i in range(m):
        row: list[float] = []
        for j in range(n):
            similarity_result = compute_similarity_func(got_list[i], expected_list[j])
            similarity = similarity_result.mean()
            row.append(similarity)
        similarity_matrix.append(row)

    # Convert to cost matrix (maximize similarity = minimize -similarity)
    cost_matrix: list[list[float]] = []
    for i in range(m):
        cost_row: list[float] = []
        for j in range(n):
            cost_row.append(-similarity_matrix[i][j])
        cost_matrix.append(cost_row)

    # Pad matrix to make it square (required by linear_sum_assignment)
    max_size = max(m, n)
    large_penalty = 1000.0
    padded_cost: list[list[float]] = []
    for i in range(max_size):
        padded_row: list[float] = []
        for j in range(max_size):
            if i < m and j < n:
                padded_row.append(cost_matrix[i][j])
            else:
                padded_row.append(large_penalty)
        padded_cost.append(padded_row)

    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(padded_cost)

    # Filter out dummy assignments
    alignment: list[tuple[int, int]] = []
    for i, j in zip(row_indices, col_indices, strict=True):
        if i < m and j < n and padded_cost[i][j] < large_penalty:
            alignment.append((i, j))

    return alignment
