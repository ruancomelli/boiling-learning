from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import timedelta
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer
from typing_extensions import ParamSpec, TypedDict

from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.descriptions import describe
from boiling_learning.io import json
from boiling_learning.io.storage import dataclass, load
from boiling_learning.lazy import Lazy, LazyDescribed
from boiling_learning.model.callbacks import RegisterEpoch, SaveHistory
from boiling_learning.model.model import Evaluation, ModelArchitecture
from boiling_learning.utils.timing import Timer
from boiling_learning.utils.typeutils import typename

_P = ParamSpec('_P')
_T = TypeVar('_T')


@describe.instance(Metric)
def _describe_typename(instance: Metric) -> str:
    return typename(instance)


class TypeAndConfig(TypedDict):
    typename: str
    config: Dict[str, Any]


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
    metrics: Optional[List[Metric]]


@dataclass(frozen=True)
class CompiledModel:
    architecture: ModelArchitecture
    params: CompileModelParams

    def compile(self) -> None:
        self.architecture.model.compile(
            optimizer=self.params.optimizer, loss=self.params.loss, metrics=self.params.metrics
        )


def compile_model(architecture: ModelArchitecture, params: CompileModelParams) -> CompiledModel:
    compiled_model = CompiledModel(architecture, params)
    compiled_model.compile()
    return compiled_model


@dataclass(frozen=True)
class FitModelParams:
    batch_size: Optional[int]
    epochs: int
    callbacks: LazyDescribed[List[Callback]]


@dataclass(frozen=True)
class FitModelReturn:
    architecture: ModelArchitecture
    trained_epochs: int
    history: Tuple[Dict[str, Any], ...]
    train_time: timedelta
    evaluation: Evaluation


def get_fit_model(
    compiled_model: CompiledModel,
    datasets: Lazy[DatasetTriplet[tf.data.Dataset]],
    params: FitModelParams,
    *,
    epoch_registry: RegisterEpoch,
    history_registry: SaveHistory,
) -> FitModelReturn:
    ds_train, ds_val, _ = datasets()

    with Timer() as timer:
        compiled_model.architecture.model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=params.epochs,
            callbacks=params.callbacks() + [history_registry, epoch_registry],
        )

    duration = timer.duration
    assert duration is not None

    return FitModelReturn(
        architecture=compiled_model.architecture,
        trained_epochs=epoch_registry.last_epoch(),
        history=tuple(history_registry.history),
        train_time=duration,
        evaluation=compiled_model.architecture.evaluate(ds_val),
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
