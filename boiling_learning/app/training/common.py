import functools
from datetime import timedelta
from typing import Any, Literal, TypedDict

import tensorflow as tf
import tensorflow_addons as tfa

from boiling_learning.app.constants import BOILING_BASELINE_BATCH_SIZE
from boiling_learning.app.paths import analyses_path
from boiling_learning.distribute import strategy_scope
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.io.dataclasses import dataclass
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import CachedFunction, Cacher
from boiling_learning.model.definitions import hoboldnet2
from boiling_learning.model.model import Evaluation, ModelArchitecture
from boiling_learning.model.training import FitModelParams, get_fit_model, load_with_strategy


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


class _CompileModelParams(TypedDict):
    loss: tf.keras.losses.Loss
    optimizer: tf.keras.optimizers.Optimizer
    metrics: list[tf.keras.metrics.Metric]


def get_baseline_compile_params(
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    learning_rate: float = 1e-3,
) -> _CompileModelParams:
    with strategy_scope(strategy):
        return {
            'loss': tf.keras.losses.MeanSquaredError(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate),
            'metrics': [
                tf.keras.metrics.MeanSquaredError('MSE'),
                tf.keras.metrics.RootMeanSquaredError('RMS'),
                tf.keras.metrics.MeanAbsoluteError('MAE'),
                tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
                tfa.metrics.RSquare('R2'),
            ],
        }


def get_baseline_fit_params(
    *,
    early_stopping_patience: int | None = 10,
) -> FitModelParams:
    return FitModelParams(
        batch_size=BOILING_BASELINE_BATCH_SIZE,
        epochs=100,
        callbacks=LazyDescribed.from_list(
            [
                LazyDescribed.from_constructor(tf.keras.callbacks.TerminateOnNaN),
                *(
                    [
                        LazyDescribed.from_constructor(
                            tf.keras.callbacks.EarlyStopping,
                            monitor='val_loss',
                            min_delta=0,
                            patience=early_stopping_patience,
                            baseline=None,
                            mode='auto',
                            restore_best_weights=True,
                            verbose=1,
                        ),
                    ]
                    if early_stopping_patience is not None
                    else []
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


@dataclass(frozen=True)
class LazyFitModelReturn:
    architecture: LazyDescribed[ModelArchitecture]
    trained_epochs: int
    history: tuple[dict[str, Any], ...]
    train_time: timedelta
    validation_metrics: Evaluation
    test_metrics: Evaluation
