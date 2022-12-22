from functools import cache
from typing import Literal, Optional

import tensorflow as tf

from boiling_learning.app.automl.tuning import autofit
from boiling_learning.app.paths import analyses_path
from boiling_learning.app.training.common import get_baseline_compile_params
from boiling_learning.automl.hypermodels import ConvImageRegressor, HyperModel
from boiling_learning.automl.tuners import EarlyStoppingGreedy
from boiling_learning.automl.tuning import TuneModelParams, TuneModelReturn
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.io.storage import dataclass
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator


@dataclass
class AutofitHypermodel:
    hypermodel: HyperModel
    tune_model_return: TuneModelReturn


def autofit_dataset(
    datasets: LazyDescribed[ImageDatasetTriplet],
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    target: str,
    experiment: Literal['boiling1d', 'condensation'],
    normalize_images: bool = True,
    max_model_size: Optional[int] = None,
    goal: float | None = None,
) -> AutofitHypermodel:
    compile_params = get_baseline_compile_params(strategy=strategy)

    hypermodel = ConvImageRegressor(
        loss=compile_params['loss'],
        metrics=compile_params['metrics'],
        tuner=EarlyStoppingGreedy,
        directory=_get_autofit_to_dataset_allocator(experiment).allocate(
            ConvImageRegressor,
            datasets,
            tuner=EarlyStoppingGreedy,
            loss=compile_params['loss'],
            metrics=compile_params['metrics'],
            normalize_images=normalize_images,
            max_model_size=max_model_size,
            goal=goal,
        ),
        max_model_size=max_model_size,
        strategy=strategy,
        normalize_images=normalize_images,
        goal=goal,
    )

    tune_model_params = TuneModelParams(
        batch_size=32,
        callbacks=LazyDescribed.from_list(
            [
                LazyDescribed.from_constructor(tf.keras.callbacks.TerminateOnNaN),
                LazyDescribed.from_constructor(
                    tf.keras.callbacks.EarlyStopping,
                    monitor='val_loss',
                    min_delta=0,
                    patience=10,
                    baseline=None,
                    mode='auto',
                    restore_best_weights=True,
                    verbose=1,
                ),
            ]
        ),
    )

    tune_model_return = autofit(
        hypermodel,
        datasets=datasets,
        params=tune_model_params,
        target=target,
        experiment=experiment,
        strategy=strategy,
    )

    return AutofitHypermodel(hypermodel=hypermodel, tune_model_return=tune_model_return)


@cache
def _get_autofit_to_dataset_allocator(
    experiment: Literal['boiling1d', 'condensation'],
    /,
) -> JSONAllocator:
    return JSONAllocator(analyses_path() / 'autofit' / 'autofit-to-dataset' / experiment)
