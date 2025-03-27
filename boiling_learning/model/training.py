from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any, ParamSpec, TypedDict, TypeVar

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer

from boiling_learning.datasets.splits import DatasetTriplet
from boiling_learning.descriptions import describe
from boiling_learning.distribute import strategy_scope
from boiling_learning.io import json
from boiling_learning.io.dataclasses import dataclass
from boiling_learning.io.storage import load
from boiling_learning.lazy import Lazy, LazyDescribed, eager
from boiling_learning.model.callbacks import RegisterEpoch, SaveHistory
from boiling_learning.model.model import Evaluation, ModelArchitecture
from boiling_learning.preprocessing.transformers import wrap_as_partial_transformer
from boiling_learning.utils.timing import Timer
from boiling_learning.utils.typeutils import typename

_P = ParamSpec("_P")
_T = TypeVar("_T")


@describe.instance(Metric)
def _describe_metric(instance: Metric) -> str:
    return typename(instance)


class TypeAndConfig(TypedDict):
    typename: str
    config: dict[str, Any]


@describe.instance(Loss)
@describe.instance(Optimizer)
def _describe_configurable(instance: Loss | Optimizer) -> TypeAndConfig:
    return {
        "typename": typename(instance),
        "config": _numpy_types_to_builtin(instance.get_config()),
    }


def _numpy_types_to_builtin(model_config: dict[str, Any]) -> dict[str, Any]:
    for key, value in model_config.items():
        if isinstance(value, dict):
            model_config[key] = _numpy_types_to_builtin(value)
        elif isinstance(value, list | tuple):
            model_config[key] = [_numpy_types_to_builtin(item) for item in value]
        elif value is None or isinstance(value, int | float | str | bool):
            pass
        elif isinstance(value, np.bool_):
            model_config[key] = bool(value)
        elif isinstance(value, np.integer):
            model_config[key] = int(value)
        elif isinstance(value, np.floating):
            model_config[key] = float(value)
        else:
            raise TypeError(
                f"Unsupported type in model config: got type {type(value)} for value {value}"
            )
    return model_config


@json.encode.instance(Loss)
@json.encode.instance(Metric)
@json.encode.instance(Optimizer)
def _encode_configurable(instance: Loss | Metric | Optimizer) -> json.JSONDataType:
    return json.serialize(describe(instance))


@wrap_as_partial_transformer
@eager
def compile_model(
    architecture: ModelArchitecture,
    /,
    *,
    loss: Loss,
    optimizer: Optimizer | str,
    metrics: list[Metric] | None,
) -> ModelArchitecture:
    # TODO: create a clone here to ensure pureness
    # needs to do the clone in the same `strategy` as metrics
    # architecture = architecture.clone()

    architecture.model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    return architecture


@dataclass(frozen=True)
class FitModelParams:
    batch_size: int | None
    epochs: int
    callbacks: LazyDescribed[list[Callback]]


@dataclass(frozen=True)
class FitModelReturn:
    architecture: ModelArchitecture
    trained_epochs: int
    history: tuple[dict[str, Any], ...]
    train_time: timedelta
    validation_metrics: Evaluation
    test_metrics: Evaluation


def get_fit_model(
    model: ModelArchitecture,
    datasets: DatasetTriplet[tf.data.Dataset],
    params: FitModelParams,
    *,
    epoch_registry: RegisterEpoch,
    history_registry: SaveHistory,
) -> FitModelReturn:
    ds_train, ds_val, ds_test = datasets

    with Timer() as timer:
        model.model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=params.epochs,
            callbacks=params.callbacks() + [history_registry, epoch_registry],
        )

    duration = timer.duration
    assert duration is not None

    return FitModelReturn(
        architecture=model,
        trained_epochs=epoch_registry.last_epoch(),
        history=tuple(history_registry.history),
        train_time=duration,
        validation_metrics=model.evaluate(ds_val),
        test_metrics=model.evaluate(ds_test),
    )


def _wrap_with_strategy(
    func: Callable[_P, _T],
) -> Callable[[Lazy[tf.distribute.Strategy] | None], Callable[_P, _T]]:
    def _wrapper(strategy: Lazy[tf.distribute.Strategy] | None) -> Callable[_P, _T]:
        def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            with strategy_scope(strategy):
                return func(*args, **kwargs)

        return _wrapped

    return _wrapper


load_with_strategy = _wrap_with_strategy(load)
