from importlib.metadata import version

from cobjectric.base_model import BaseModel
from cobjectric.exceptions import (
    CobjectricError,
    DuplicateFillRateFuncError,
    InvalidFillRateValueError,
    InvalidWeightError,
    MissingListTypeArgError,
    UnsupportedListTypeError,
    UnsupportedTypeError,
)
from cobjectric.field_spec import FieldSpec, Spec
from cobjectric.fill_rate import (
    FillRateFieldResult,
    FillRateFuncInfo,
    FillRateModelResult,
    fill_rate_func,
)
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
    "DuplicateFillRateFuncError",
    "InvalidFillRateValueError",
    "InvalidWeightError",
    "FieldSpec",
    "Spec",
    "field_normalizer",
    "fill_rate_func",
    "FillRateFieldResult",
    "FillRateFuncInfo",
    "FillRateModelResult",
]
