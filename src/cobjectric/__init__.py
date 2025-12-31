from importlib.metadata import version

from cobjectric.base_model import BaseModel
from cobjectric.exceptions import (
    CobjectricError,
    MissingListTypeArgError,
    UnsupportedListTypeError,
    UnsupportedTypeError,
)
from cobjectric.field_spec import FieldSpec, Spec
from cobjectric.sentinel import MissingValue

__version__ = version("cobjectric")

__all__ = [
    "BaseModel",
    "MissingValue",
    "CobjectricError",
    "UnsupportedListTypeError",
    "MissingListTypeArgError",
    "UnsupportedTypeError",
    "FieldSpec",
    "Spec",
]


def status() -> bool:
    """
    Check if the library is working

    Returns:
        bool: True if the library is working, False otherwise
    """
    return True
