from typing_extensions import Sentinel

from cobjectric import MissingValue


def test_missing_value_is_sentinel() -> None:
    """Test that MissingValue is a Sentinel instance."""
    assert isinstance(MissingValue, Sentinel)


def test_missing_value_repr() -> None:
    """Test that MissingValue has a proper representation."""
    assert repr(MissingValue) == "<MissingValue>"


def test_missing_value_is_singleton() -> None:
    """Test that MissingValue is a singleton."""
    from cobjectric.sentinel import MissingValue as MissingValue2

    assert MissingValue is MissingValue2


def test_missing_value_can_be_imported() -> None:
    """Test that MissingValue can be imported from cobjectric."""
    from cobjectric import MissingValue

    assert MissingValue is not None
