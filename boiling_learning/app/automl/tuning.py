from dataclasses import replace
from typing import Literal

import tensorflow as tf

from boiling_learning.app.datasets.bridging import to_tensorflow_triplet
from boiling_learning.app.paths import analyses_path
from boiling_learning.automl.hypermodels import HyperModel
from boiling_learning.automl.tuning import TuneModelParams, TuneModelReturn, fit_hypermodel
from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import CachedFunction, cache
from boiling_learning.model.callbacks import MemoryCleanUp
from boiling_learning.model.model import rename_model_layers
from boiling_learning.model.training import load_with_strategy


def autofit(
    hypermodel: HyperModel,
    datasets: LazyDescribed[ImageDatasetTriplet],
    params: TuneModelParams,
    target: str,
    experiment: Literal['boiling1d', 'condensation'],
    strategy: LazyDescribed[tf.distribute.Strategy],
) -> TuneModelReturn:
    return _cached_autofit_function(experiment=experiment, strategy=strategy,)(
        hypermodel,
        datasets,
        params,
        target=target,
    )


def _cached_autofit_function(
    experiment: Literal['boiling1d', 'condensation'],
    strategy: LazyDescribed[tf.distribute.Strategy],
) -> CachedFunction:
    @cache(
        allocator=JSONAllocator(analyses_path() / 'autofit' / experiment),
        exceptions=(FileNotFoundError, NotADirectoryError, tf.errors.OpError),
        loader=load_with_strategy(strategy),
    )
    def autofit(
        hypermodel: HyperModel,
        datasets: LazyDescribed[ImageDatasetTriplet],
        params: TuneModelParams,
        target: str,
    ) -> TuneModelReturn:
        ds_train, ds_val, ds_test = to_tensorflow_triplet(
            datasets,
            batch_size=params.batch_size,
            include_test=False,
            target=target,
            experiment=experiment,
        )

        if not any(isinstance(callback, MemoryCleanUp) for callback in params.callbacks()):
            params.callbacks().append(MemoryCleanUp())

        tuned_model = fit_hypermodel(
            hypermodel,
            DatasetTriplet(
                ds_train().unbatch().prefetch(tf.data.AUTOTUNE),
                ds_val().unbatch().prefetch(tf.data.AUTOTUNE),
                ds_test(),
            ),
            params,
        )

        tuned_model = replace(tuned_model, model=rename_model_layers(tuned_model.model))

        return tuned_model

    return autofit
