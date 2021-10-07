from typing import Callable, Dict, Iterable, Tuple

import tensorflow as tf

from boiling_learning.datasets.datasets import features, targets
from boiling_learning.datasets.metrics import calculate_metric

SingleMetricEvaluator = Callable[[tf.keras.Model], float]
MultiMetricEvaluator = Callable[[tf.keras.Model], Dict[str, float]]


class DatasetMultiMetricEvaluator:
    def __init__(
        self,
        dataset: tf.data.Dataset,
        metrics: Iterable[tf.keras.metrics.Metric],
    ) -> None:
        self.features: tf.data.Dataset = features(dataset)
        self.targets: tf.data.Dataset = targets(dataset)
        self.metrics: Tuple[tf.keras.metrics.Metric, ...] = tuple(metrics)

    def __call__(self, model: tf.keras.Model) -> Dict[str, float]:
        y_pred = model.predict(self.features)
        zipped = tf.data.Dataset.zip(self.targets, y_pred)

        return {
            metric.name: calculate_metric(zipped, metric)
            for metric in self.metrics
        }
