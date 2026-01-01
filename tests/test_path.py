import pytest

from cobjectric.path import parse_path


def test_parse_path_simple() -> None:
    """Test parsing simple paths."""
    segments = parse_path("name")
    assert segments == ["name"]


def test_parse_path_with_brackets() -> None:
    """Test parsing paths with list index brackets."""
    segments = parse_path("items[0].name")
    assert segments == ["items", "[0]", "name"]


def test_parse_path_nested() -> None:
    """Test parsing nested paths."""
    segments = parse_path("address.city")
    assert segments == ["address", "city"]


def test_parse_path_complex() -> None:
    """Test parsing complex paths with multiple levels and indices."""
    segments = parse_path("items[0].address.street")
    assert segments == ["items", "[0]", "address", "street"]


def test_parse_path_multiple_indices() -> None:
    """Test parsing paths with multiple list indices."""
    segments = parse_path("items[0].tags[1]")
    assert segments == ["items", "[0]", "tags", "[1]"]


def test_parse_path_invalid_unclosed_bracket() -> None:
    """Test parsing path with unclosed bracket raises KeyError."""
    with pytest.raises(KeyError, match="Invalid path"):
        _ = parse_path("name[0")


def test_parse_path_invalid_non_numeric_bracket() -> None:
    """Test parsing path with non-numeric bracket raises KeyError."""
    with pytest.raises(KeyError, match="Invalid path"):
        _ = parse_path("name[abc]")


def test_parse_path_empty() -> None:
    """Test parsing empty path."""
    segments = parse_path("")
    assert segments == []


def test_parse_path_only_index() -> None:
    """Test parsing path that starts with index (should work but unusual)."""
    segments = parse_path("[0]")
    assert segments == ["[0]"]
