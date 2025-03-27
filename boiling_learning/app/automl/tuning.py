from dataclasses import replace
from typing import Literal

import tensorflow as tf

from boiling_learning.app.datasets.bridging import to_tensorflow_triplet
from boiling_learning.automl.hypermodels import HyperModel
from boiling_learning.automl.tuning import (
    TuneModelParams,
    TuneModelReturn,
    fit_hypermodel,
)
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.callbacks import MemoryCleanUp
from boiling_learning.model.model import rename_model_layers


def autofit(
    hypermodel: HyperModel,
    datasets: LazyDescribed[ImageDatasetTriplet],
    params: TuneModelParams,
    target: str,
    experiment: Literal["boiling1d", "condensation"],
    strategy: LazyDescribed[tf.distribute.Strategy],
) -> TuneModelReturn:
    datasets_tf = to_tensorflow_triplet(
        datasets,
        batch_size=None,
        target=target,
        experiment=experiment,
    )

    if not any(isinstance(callback, MemoryCleanUp) for callback in params.callbacks()):
        params.callbacks().append(MemoryCleanUp())

    tuned_model = fit_hypermodel(
        hypermodel,
        datasets_tf,
        params,
    )

    tuned_model = replace(
        tuned_model,
        model=rename_model_layers(tuned_model.model, strategy=strategy),
    )

    return tuned_model
