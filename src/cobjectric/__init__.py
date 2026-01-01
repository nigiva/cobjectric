from importlib.metadata import version

from cobjectric.base_model import BaseModel
from cobjectric.exceptions import (
    CobjectricError,
    DuplicateFillRateAccuracyFuncError,
    DuplicateFillRateFuncError,
    DuplicateSimilarityFuncError,
    InvalidAggregatedFieldError,
    InvalidFillRateValueError,
    InvalidWeightError,
    MissingListTypeArgError,
    UnsupportedListTypeError,
    UnsupportedTypeError,
)
from cobjectric.field_spec import FieldSpec, Spec
from cobjectric.fill_rate import (
    FillRateAccuracyFuncInfo,
    FillRateAggregatedFieldCollection,
    FillRateAggregatedFieldResult,
    FillRateAggregatedModelResult,
    FillRateFieldResult,
    FillRateFuncInfo,
    FillRateListResult,
    FillRateModelResult,
    FillRateNestedListAggregatedResult,
    SimilarityFuncInfo,
    fill_rate_accuracy_func,
    fill_rate_func,
    not_missing_fill_rate,
    same_state_fill_rate_accuracy,
    similarity_func,
)
from cobjectric.normalizer import field_normalizer
from cobjectric.sentinel import MissingValue
from cobjectric.similarities import (
    exact_similarity,
    fuzzy_similarity_factory,
    numeric_similarity_factory,
)

__version__ = version("cobjectric")

__all__ = [
    "BaseModel",
    "MissingValue",
    "CobjectricError",
    "UnsupportedListTypeError",
    "MissingListTypeArgError",
    "UnsupportedTypeError",
    "DuplicateFillRateFuncError",
    "DuplicateFillRateAccuracyFuncError",
    "DuplicateSimilarityFuncError",
    "InvalidFillRateValueError",
    "InvalidWeightError",
    "InvalidAggregatedFieldError",
    "FieldSpec",
    "Spec",
    "field_normalizer",
    "fill_rate_func",
    "fill_rate_accuracy_func",
    "similarity_func",
    "FillRateFieldResult",
    "FillRateFuncInfo",
    "FillRateAccuracyFuncInfo",
    "SimilarityFuncInfo",
    "FillRateModelResult",
    "FillRateListResult",
    "FillRateAggregatedFieldResult",
    "FillRateAggregatedModelResult",
    "FillRateAggregatedFieldCollection",
    "FillRateNestedListAggregatedResult",
    "exact_similarity",
    "fuzzy_similarity_factory",
    "numeric_similarity_factory",
    "not_missing_fill_rate",
    "same_state_fill_rate_accuracy",
]
