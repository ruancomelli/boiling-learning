from __future__ import annotations

from math import floor, log10
from typing import Any, Literal, TypeAlias

import numpy as np
import tensorflow as tf
from scipy.stats import bootstrap

from boiling_learning.io.dataclasses import dataclass
from boiling_learning.model.model import ModelArchitecture

MetricName: TypeAlias = str


@dataclass(frozen=True, order=True)
class UncertainValue:
    value: float
    upper: float
    lower: float

    def __str__(self) -> str:
        return f'{self.value} + {self.upper} - {self.lower}'

    def rounded(self) -> UncertainValue:
        position_to_round = min(
            _position_of_most_significant_digit(self.upper),
            _position_of_most_significant_digit(self.lower),
        )

        rounded_mean = round(self.value, position_to_round)
        rounded_upper = round(self.upper, position_to_round)
        rounded_lower = round(self.lower, position_to_round)

        return UncertainValue(
            rounded_mean,
            rounded_upper,
            rounded_lower,
        )


UncertainEvaluation: TypeAlias = dict[MetricName, UncertainValue]


def evaluate_with_uncertainty(
    model: ModelArchitecture,
    dataset: tf.data.Dataset,
    *,
    batches: int = 1000,
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

    mean = float(np.mean(histogram))
    return UncertainValue(
        mean,
        lower=mean - float(confidence_interval.low),
        upper=float(confidence_interval.high) - mean,
    )


def _position_of_most_significant_digit(x: float, /) -> int:
    return -int(floor(log10(abs(x))))
