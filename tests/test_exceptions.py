import pytest

from cobjectric import CobjectricError


def test_cobjectric_error_inherits_from_exception() -> None:
    """Test that CobjectricError inherits from Exception."""
    assert issubclass(CobjectricError, Exception)


def test_cobjectric_error_can_be_instantiated() -> None:
    """Test that CobjectricError can be instantiated."""
    error = CobjectricError()
    assert isinstance(error, Exception)
    assert isinstance(error, CobjectricError)


def test_cobjectric_error_with_message() -> None:
    """Test that CobjectricError can be instantiated with a message."""
    message = "Test error message"
    error = CobjectricError(message)
    assert str(error) == message


def test_cobjectric_error_can_be_raised() -> None:
    """Test that CobjectricError can be raised and caught."""
    with pytest.raises(CobjectricError) as exc_info:
        raise CobjectricError("Test error")
    assert str(exc_info.value) == "Test error"
