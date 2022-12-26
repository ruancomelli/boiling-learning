from numbers import Real
from typing import Any, Generic, Literal, TypeAlias, TypeVar

import numpy as np
import tensorflow as tf
from scipy.stats import bootstrap

from boiling_learning.io.storage import dataclass
from boiling_learning.model.model import ModelArchitecture

_T = TypeVar('_T', bound=Real)
MetricName: TypeAlias = str


@dataclass
class UncertainValue(Generic[_T]):
    value: _T
    upper: _T
    lower: _T

    def __str__(self) -> str:
        upper_diff = self.upper - self.value
        lower_diff = self.value - self.lower
        return f'{self.value} + {upper_diff} - {lower_diff}'


def evaluate_with_uncertainty(
    model: ModelArchitecture,
    dataset: tf.data.Dataset,
    *,
    bins: int = 100,
    method: Literal['percentile', 'basic', 'bca'] = 'bca',
    confidence_level: float = 0.95,
) -> dict[str, UncertainValue]:
    histograms: dict[MetricName, list[Any]] = {}
    for x, y in dataset.batch(bins):
        evaluation = model.evaluate(x, y)
        for metric_name, metric_value in evaluation.items():
            histograms.setdefault(metric_name, []).append(metric_value)

    return {
        key: _uncertain_value_from_histogram(
            histogram,
            method=method,
            confidence_level=confidence_level,
            bins=bins,
        )
        for key, histogram in histograms.items()
    }


def _uncertain_value_from_histogram(
    histogram: list[_T],
    method: Literal['percentile', 'basic', 'bca'] = 'bca',
    confidence_level: float = 0.95,
    bins: int = 100,
) -> UncertainValue[_T]:
    confidence_interval = bootstrap(
        (histogram,),
        np.mean,
        method=method,
        confidence_level=confidence_level,
        n_resamples=bins,
    ).confidence_interval

    return UncertainValue(
        np.mean(histogram),
        lower=confidence_interval.low,
        upper=confidence_interval.high,
    )
