from importlib.metadata import version

from cobjectric.base_model import BaseModel
from cobjectric.context import FieldContext
from cobjectric.exceptions import (
    CobjectricError,
    DuplicateFillRateAccuracyFuncError,
    DuplicateFillRateFuncError,
    DuplicateSimilarityFuncError,
    IncompatibleModelResultError,
    InvalidAggregatedFieldError,
    InvalidFillRateValueError,
    InvalidListCompareStrategyError,
    InvalidWeightError,
    MissingListTypeArgError,
    UnsupportedListTypeError,
    UnsupportedTypeError,
)
from cobjectric.field_spec import FieldSpec, Spec
from cobjectric.fill_rate import (
    FillRateAccuracyFuncInfo,
    FillRateFuncInfo,
    SimilarityFuncInfo,
    fill_rate_accuracy_func,
    fill_rate_func,
    not_missing_fill_rate,
    same_state_fill_rate_accuracy,
    similarity_func,
)
from cobjectric.list_compare import ListCompareStrategy
from cobjectric.normalizer import field_normalizer
from cobjectric.results import (
    AggregatedFieldResult,
    AggregatedFieldResultCollection,
    AggregatedModelResult,
    FieldResult,
    FieldResultCollection,
    ListResult,
    ModelResult,
    ModelResultCollection,
    NestedListAggregatedResult,
)
from cobjectric.sentinel import MissingValue
from cobjectric.similarities import (
    datetime_similarity_factory,
    exact_similarity,
    fuzzy_similarity_factory,
    numeric_similarity_factory,
)
from cobjectric.specs import (
    BooleanSpec,
    DatetimeSpec,
    KeywordSpec,
    NumericSpec,
    TextSpec,
)

__version__ = version("cobjectric")

__all__ = [
    "BaseModel",
    "FieldContext",
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
    "InvalidListCompareStrategyError",
    "IncompatibleModelResultError",
    "FieldSpec",
    "Spec",
    "field_normalizer",
    "fill_rate_func",
    "fill_rate_accuracy_func",
    "similarity_func",
    "FieldResult",
    "FieldResultCollection",
    "FillRateFuncInfo",
    "FillRateAccuracyFuncInfo",
    "SimilarityFuncInfo",
    "ModelResult",
    "ModelResultCollection",
    "ListResult",
    "AggregatedFieldResult",
    "AggregatedModelResult",
    "AggregatedFieldResultCollection",
    "NestedListAggregatedResult",
    "ListCompareStrategy",
    "exact_similarity",
    "fuzzy_similarity_factory",
    "numeric_similarity_factory",
    "datetime_similarity_factory",
    "not_missing_fill_rate",
    "same_state_fill_rate_accuracy",
    "KeywordSpec",
    "TextSpec",
    "NumericSpec",
    "BooleanSpec",
    "DatetimeSpec",
]
