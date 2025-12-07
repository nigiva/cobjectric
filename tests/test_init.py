from cobjectric import status


def test_status_returns_true() -> None:
    """Test that status() returns True when the library is working."""
    result = status()

    assert result is True


def test_status_returns_bool() -> None:
    """Test that status() returns a boolean type."""
    result = status()

    assert isinstance(result, bool)
