from importlib.metadata import version

from cobjectric.base_model import BaseModel
from cobjectric.exceptions import (
    CobjectricError,
    MissingListTypeArgError,
    UnsupportedListTypeError,
    UnsupportedTypeError,
)
from cobjectric.field_spec import FieldSpec, Spec
from cobjectric.normalizer import field_normalizer
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
    "field_normalizer",
]
