from functools import cache
from typing import Literal

import tensorflow as tf

from boiling_learning.app.automl.evaluation import cached_best_model_evaluator
from boiling_learning.app.automl.tuning import autofit
from boiling_learning.app.datasets.preprocessed.boiling1d import (
    baseline_boiling_dataset,
)
from boiling_learning.app.paths import analyses_path
from boiling_learning.app.training.common import get_baseline_compile_params
from boiling_learning.automl.hypermodels import ConvImageRegressor, HyperModel
from boiling_learning.automl.tuners import AutoTuner, EarlyStoppingGreedy
from boiling_learning.automl.tuning import TuneModelParams
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.model.model import ModelArchitecture


def autofit_dataset(
    datasets: LazyDescribed[ImageDatasetTriplet],
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    target: str,
    experiment: Literal["boiling1d", "condensation"],
    normalize_images: bool = True,
    max_model_size: int | None = None,
    goal: float | None = None,
    tuner_class: type[AutoTuner] = EarlyStoppingGreedy,
) -> HyperModel:
    compile_params = get_baseline_compile_params(strategy=strategy)

    hypermodel = ConvImageRegressor(
        loss=compile_params["loss"],
        metrics=compile_params["metrics"],
        tuner=tuner_class,
        directory=_get_autofit_to_dataset_allocator(experiment).allocate(
            ConvImageRegressor,
            datasets,
            tuner=tuner_class,
            loss=compile_params["loss"],
            metrics=compile_params["metrics"],
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
                    monitor="val_loss",
                    min_delta=0,
                    patience=10,
                    baseline=None,
                    mode="auto",
                    restore_best_weights=True,
                    verbose=1,
                ),
            ]
        ),
    )

    autofit(
        hypermodel,
        datasets=datasets,
        params=tune_model_params,
        target=target,
        experiment=experiment,
        strategy=strategy,
    )

    return hypermodel


def best_model_for_dataset(
    datasets: LazyDescribed[ImageDatasetTriplet],
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    target: str,
    experiment: Literal["boiling1d", "condensation"],
    normalize_images: bool = True,
    max_model_size: int | None = None,
    goal: float | None = None,
    tuner_class: type[AutoTuner] = EarlyStoppingGreedy,
) -> LazyDescribed[ModelArchitecture]:
    hypermodel = autofit_dataset(
        datasets,
        strategy=strategy,
        target=target,
        experiment=experiment,
        normalize_images=normalize_images,
        max_model_size=max_model_size,
        goal=goal,
        tuner_class=tuner_class,
    )
    best_model_evaluator = cached_best_model_evaluator(
        experiment="boiling1d",
        strategy=strategy,
    )
    return best_model_evaluator(
        hypermodel,
        datasets,
        measure_uncertainty=False,
    )


def best_baseline_boiling1d_model(
    *,
    direct_visualization: bool,
    strategy: LazyDescribed[tf.distribute.Strategy],
    target: str,
    normalize_images: bool = True,
    max_model_size: int | None = None,
    goal: float | None = None,
    tuner_class: type[AutoTuner] = EarlyStoppingGreedy,
) -> LazyDescribed[ModelArchitecture]:
    datasets = baseline_boiling_dataset(direct_visualization=direct_visualization)
    return best_model_for_dataset(
        datasets,
        strategy=strategy,
        target=target,
        experiment="boiling1d",
        normalize_images=normalize_images,
        max_model_size=max_model_size,
        goal=goal,
        tuner_class=tuner_class,
    )


@cache
def _get_autofit_to_dataset_allocator(
    experiment: Literal["boiling1d", "condensation"],
    /,
) -> JSONAllocator:
    return JSONAllocator(
        analyses_path() / "autofit" / "autofit-to-dataset" / experiment
    )
