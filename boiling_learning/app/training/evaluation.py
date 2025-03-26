import functools
from collections.abc import Iterator
from typing import Literal

import tensorflow as tf

from boiling_learning.app.datasets.bridged.boiling1d import (
    default_boiling_bridging,
    default_boiling_bridging_gt10,
)
from boiling_learning.app.paths import analyses_path
from boiling_learning.datasets.splits import DatasetTriplet
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.io.dataclasses import dataclass
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.model.evaluate import (
    UncertainEvaluation,
    UncertainValue,
    evaluate_with_uncertainty,
)
from boiling_learning.model.model import ModelArchitecture

EVALUATION_BATCH_SIZE = 32


@dataclass(frozen=True)
class ModelEvaluation:
    trainable_parameters_count: int
    total_parameters_count: int

    training_metrics: UncertainEvaluation
    validation_metrics: UncertainEvaluation
    test_metrics: UncertainEvaluation

    @property
    def metrics_names(self) -> list[str]:
        return sorted(
            self.training_metrics,
            key=lambda metric_name: {
                "loss": 0,
                "MSE": 1,
                "RMS": 2,
                "MAE": 3,
                "MAPE": 4,
                "R2": 5,
            }.get(metric_name, 10),
        )

    def iter_metrics(
        self,
    ) -> Iterator[tuple[str, UncertainValue, UncertainValue, UncertainValue]]:
        return (
            (
                metric_name,
                self.training_metrics[metric_name],
                self.validation_metrics[metric_name],
                self.test_metrics[metric_name],
            )
            for metric_name in self.metrics_names
        )


@functools.cache
def cached_model_evaluator(
    experiment: Literal["boiling1d", "condensation"],
    /,
):
    @cache(JSONAllocator(analyses_path() / "evaluations" / experiment))
    def model_evaluator(
        model: LazyDescribed[ModelArchitecture],
        datasets: LazyDescribed[ImageDatasetTriplet],
        *,
        measure_uncertainty: bool = True,
        gt10: bool = True,
    ) -> ModelEvaluation:
        training_metrics, validation_metrics, test_metrics = (
            evaluate_boiling_model_with_dataset(
                model,
                datasets,
                measure_uncertainty=measure_uncertainty,
                gt10=gt10,
            )
        )

        trainable_size = model().count_parameters(
            trainable=True,
            non_trainable=False,
        )
        total_size = model().count_parameters(
            trainable=True,
            non_trainable=True,
        )

        return ModelEvaluation(
            trainable_parameters_count=trainable_size,
            total_parameters_count=total_size,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
        )

    return model_evaluator


def evaluate_boiling_model_with_dataset(
    model: LazyDescribed[ModelArchitecture],
    evaluation_dataset: LazyDescribed[ImageDatasetTriplet],
    *,
    measure_uncertainty: bool = True,
    gt10: bool = True,
) -> DatasetTriplet[dict[str, UncertainValue]]:
    bridger = default_boiling_bridging_gt10 if gt10 else default_boiling_bridging
    ds_train, ds_val, ds_test = bridger(
        evaluation_dataset,
        batch_size=None,
    )

    if measure_uncertainty:
        train_metrics = evaluate_with_uncertainty(model(), ds_train())
        validation_metrics = evaluate_with_uncertainty(model(), ds_val())
        test_metrics = evaluate_with_uncertainty(model(), ds_test())
    else:
        train_metrics = model().evaluate(
            ds_train().batch(EVALUATION_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        )
        validation_metrics = model().evaluate(
            ds_val().batch(EVALUATION_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        )
        test_metrics = model().evaluate(
            ds_test().batch(EVALUATION_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        )

    return DatasetTriplet(train_metrics, validation_metrics, test_metrics)
