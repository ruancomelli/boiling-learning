from typing import Any, Literal, TypeAlias

import numpy as np
import tensorflow as tf
from scipy.stats import bootstrap

from boiling_learning.io.storage import dataclass
from boiling_learning.model.model import ModelArchitecture

MetricName: TypeAlias = str


@dataclass
class UncertainValue:
    value: float
    upper: float
    lower: float

    def __str__(self) -> str:
        upper_diff = self.upper - self.value
        lower_diff = self.value - self.lower
        return f'{self.value} + {upper_diff} - {lower_diff}'


UncertainEvaluation: TypeAlias = dict[MetricName, UncertainValue]


def evaluate_with_uncertainty(
    model: ModelArchitecture,
    dataset: tf.data.Dataset,
    *,
    batches: int = 10000,
    samples: int = 1000,
    method: Literal['percentile', 'basic', 'bca'] = 'bca',
    confidence_level: float = 0.95,
) -> UncertainEvaluation:
    histograms: dict[MetricName, list[Any]] = {}
    for x, y in dataset.repeat().batch(samples).take(batches):
        evaluation = model.evaluate(x, y)
        for metric_name, metric_value in evaluation.items():
            histograms.setdefault(metric_name, []).append(float(metric_value))

    return {
        key: _uncertain_value_from_histogram(
            histogram,
            method=method,
            confidence_level=confidence_level,
        )
        for key, histogram in histograms.items()
    }


def _uncertain_value_from_histogram(
    histogram: list[float],
    method: Literal['percentile', 'basic', 'bca'] = 'bca',
    confidence_level: float = 0.95,
) -> UncertainValue:
    confidence_interval = bootstrap(
        (histogram,),
        np.mean,
        method=method,
        confidence_level=confidence_level,
    ).confidence_interval

    return UncertainValue(
        float(np.mean(histogram)),
        lower=float(confidence_interval.low),
        upper=float(confidence_interval.high),
    )
