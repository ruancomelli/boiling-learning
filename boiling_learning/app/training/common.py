import functools
from typing import Literal

import tensorflow as tf
import tensorflow_addons as tfa

from boiling_learning.app.constants import BOILING_BASELINE_BATCH_SIZE
from boiling_learning.app.paths import analyses_path
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import CachedFunction, Cacher
from boiling_learning.model.definitions import hoboldnet2
from boiling_learning.model.model import ModelArchitecture
from boiling_learning.model.training import (
    CompileModelParams,
    FitModelParams,
    get_fit_model,
    load_with_strategy,
    strategy_scope,
)


@functools.cache
def cached_fit_model_function(
    experiment: Literal['boiling1d', 'condensation'],
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
) -> CachedFunction:
    return CachedFunction(
        get_fit_model,
        Cacher(
            allocator=JSONAllocator(analyses_path() / 'models' / experiment),
            exceptions=(FileNotFoundError, NotADirectoryError, tf.errors.OpError),
            loader=load_with_strategy(strategy),
        ),
    )


def get_baseline_compile_params(
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
) -> CompileModelParams:
    with strategy_scope(strategy):
        return CompileModelParams(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(1e-3),
            metrics=[
                tf.keras.metrics.MeanSquaredError('MSE'),
                tf.keras.metrics.RootMeanSquaredError('RMS'),
                tf.keras.metrics.MeanAbsoluteError('MAE'),
                tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
                tfa.metrics.RSquare('R2'),
            ],
        )


def get_baseline_fit_params() -> FitModelParams:
    return FitModelParams(
        batch_size=BOILING_BASELINE_BATCH_SIZE,
        epochs=100,
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


def get_baseline_architecture(
    dataset: LazyDescribed[ImageDatasetTriplet],
    /,
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    normalize_images: bool = True,
) -> LazyDescribed[ModelArchitecture]:
    ds_train, _, _ = dataset()
    first_frame, _ = ds_train[0]

    with strategy_scope(strategy):
        return LazyDescribed.from_describable(
            hoboldnet2(
                first_frame.shape,
                dropout=0.5,
                normalize_images=normalize_images,
            )
        )
