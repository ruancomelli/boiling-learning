import functools
from typing import Literal

import tensorflow as tf

from boiling_learning.app.paths import shared_cache_path
from boiling_learning.app.training.evaluation import cached_model_evaluator
from boiling_learning.automl.hypermodels import HyperModel
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.model.model import ModelArchitecture
from boiling_learning.model.training import load_with_strategy


@functools.cache
def cached_best_model_evaluator(
    *,
    experiment: Literal['boiling1d', 'condensation'],
    strategy: LazyDescribed[tf.distribute.Strategy],
):
    model_evaluator = cached_model_evaluator(experiment)

    @cache(
        JSONAllocator(shared_cache_path() / 'best-model-evaluations' / experiment),
        loader=load_with_strategy(strategy),
    )
    def hypermodel_evaluator(
        hypermodel: HyperModel,
        datasets: LazyDescribed[ImageDatasetTriplet],
        *,
        measure_uncertainty: bool = True,
        gt10: bool = True,
        metric_name: str = 'loss',
    ) -> LazyDescribed[ModelArchitecture]:
        return min(
            map(LazyDescribed.from_describable, hypermodel.iter_best_models()),
            key=lambda model: model_evaluator(
                model,
                datasets,
                measure_uncertainty=measure_uncertainty,
                gt10=gt10,
            ).validation_metrics[metric_name],
        )

    return hypermodel_evaluator
