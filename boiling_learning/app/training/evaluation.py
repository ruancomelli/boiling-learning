import functools
from collections.abc import Callable
from typing import Literal

from boiling_learning.app.datasets.bridged.boiling1d import default_boiling_bridging_gt10
from boiling_learning.app.paths import shared_cache_path
from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.io.storage import dataclass
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.model.evaluate import (
    UncertainEvaluation,
    UncertainValue,
    evaluate_with_uncertainty,
)
from boiling_learning.model.model import ModelArchitecture


@dataclass
class ModelEvaluation:
    trainable_parameters_count: int
    total_parameters_count: int

    training_metrics: UncertainEvaluation
    validation_metrics: UncertainEvaluation
    test_metrics: UncertainEvaluation

    @property
    def metrics_names(self) -> tuple[str, ...]:
        return tuple(self.training_metrics)


@functools.cache
def cached_model_evaluator(
    experiment: Literal['boiling1d', 'condensation'],
    /,
) -> Callable[
    [LazyDescribed[ModelArchitecture], LazyDescribed[ImageDatasetTriplet]],
    ModelEvaluation,
]:
    @cache(JSONAllocator(shared_cache_path() / experiment))
    def model_evaluator(
        model: LazyDescribed[ModelArchitecture], datasets: LazyDescribed[ImageDatasetTriplet]
    ) -> ModelEvaluation:
        training_metrics, validation_metrics, test_metrics = evaluate_boiling_model_with_dataset(
            model,
            datasets,
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
) -> DatasetTriplet[dict[str, UncertainValue]]:
    ds_train, ds_val, ds_test = default_boiling_bridging_gt10(
        evaluation_dataset,
        batch_size=None,
    )

    train_metrics = evaluate_with_uncertainty(model(), ds_train())
    validation_metrics = evaluate_with_uncertainty(model(), ds_val())
    test_metrics = evaluate_with_uncertainty(model(), ds_test())

    return DatasetTriplet(train_metrics, validation_metrics, test_metrics)
