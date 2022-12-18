from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import timedelta
from typing import Any, Callable, Iterator, Optional, ParamSpec, TypedDict, TypeVar, Union

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer

from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.descriptions import describe
from boiling_learning.io import json
from boiling_learning.io.storage import dataclass, load
from boiling_learning.lazy import Lazy, LazyDescribed, eager
from boiling_learning.model.callbacks import RegisterEpoch, SaveHistory
from boiling_learning.model.model import Evaluation, ModelArchitecture
from boiling_learning.preprocessing.transformers import wrap_as_partial_transformer
from boiling_learning.utils.timing import Timer
from boiling_learning.utils.typeutils import typename

_P = ParamSpec('_P')
_T = TypeVar('_T')


@describe.instance(Metric)
def _describe_typename(instance: Metric) -> str:
    return typename(instance)


class TypeAndConfig(TypedDict):
    typename: str
    config: dict[str, Any]


@describe.instance(Loss)
@describe.instance(Optimizer)
def _describe_configurable(instance: Union[Loss, Optimizer]) -> TypeAndConfig:
    return {'typename': typename(instance), 'config': instance.get_config()}


@json.encode.instance(Loss)
@json.encode.instance(Metric)
@json.encode.instance(Optimizer)
def _encode_configurable(instance: Union[Loss, Metric, Optimizer]) -> json.JSONDataType:
    return json.serialize(describe(instance))


@dataclass(frozen=True)
class CompileModelParams:
    loss: Loss
    optimizer: Optimizer
    metrics: Optional[list[Metric]]


@wrap_as_partial_transformer
@eager
def compile_model(
    architecture: ModelArchitecture,
    params: CompileModelParams,
) -> ModelArchitecture:
    cloned = architecture.clone()
    cloned.model.compile(
        optimizer=params.optimizer,
        loss=params.loss,
        metrics=params.metrics,
    )
    return cloned


@dataclass(frozen=True)
class FitModelParams:
    batch_size: Optional[int]
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


@contextmanager
def strategy_scope(strategy: Optional[Lazy[tf.distribute.Strategy]]) -> Iterator[None]:
    context = strategy().scope() if strategy is not None else nullcontext()

    with context:
        yield


def _wrap_with_strategy(
    func: Callable[_P, _T]
) -> Callable[[Optional[Lazy[tf.distribute.Strategy]]], Callable[_P, _T]]:
    def _wrapper(strategy: Optional[Lazy[tf.distribute.Strategy]]) -> Callable[_P, _T]:
        def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            with strategy_scope(strategy):
                return func(*args, **kwargs)

        return _wrapped

    return _wrapper


load_with_strategy = _wrap_with_strategy(load)
