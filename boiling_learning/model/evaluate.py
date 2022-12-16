from typing import Any, Generic, Literal, TypeAlias, TypeVar

import more_itertools as mit
import numpy as np
import tensorflow as tf
from scipy.stats import bootstrap

from boiling_learning.io.storage import dataclass
from boiling_learning.model.model import ModelArchitecture

_T = TypeVar('_T')
MetricName: TypeAlias = str


@dataclass
class UncertainValue(Generic[_T]):
    value: _T
    upper: _T
    lower: _T


def evaluate_with_uncertainty(
    model: ModelArchitecture,
    dataset: tf.data.Dataset,
    *,
    bins: int = 100,
    method: Literal['percentile', 'basic', 'bca'] = 'bca',
    confidence_level: float = 0.95,
) -> dict[str, UncertainValue]:
    histograms: dict[MetricName, list[Any]] = {}
    for batch in mit.batched(dataset, bins):
        evaluation = model.evaluate(batch)
        for key, value in evaluation.items():
            histograms.setdefault(key, []).append(value)

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
        histogram,
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
