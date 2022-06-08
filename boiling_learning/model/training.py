from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer
from typing_extensions import TypedDict

from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.io import json
from boiling_learning.io.storage import Metadata, deserialize, load, save, serialize
from boiling_learning.model.callbacks import RegisterEpoch, SaveHistory
from boiling_learning.model.model import ModelArchitecture
from boiling_learning.utils.dataclasses import dataclass, fields, shallow_asdict
from boiling_learning.utils.described import Described
from boiling_learning.utils.descriptions import describe
from boiling_learning.utils.timing import Timer
from boiling_learning.utils.typeutils import typename
from boiling_learning.utils.utils import resolve


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


def compile_model(architecture: ModelArchitecture, params: CompileModelParams) -> CompiledModel:
    architecture.model.compile(
        optimizer=params.optimizer, loss=params.loss, metrics=params.metrics
    )
    return CompiledModel(architecture, params)


@dataclass(frozen=True)
class FitModelParams:
    batch_size: Optional[int]
    epochs: int
    callbacks: Described[List[Callback], json.JSONDataType]


@dataclass(frozen=True)
class FitModelReturn:
    architecture: ModelArchitecture
    trained_epochs: int
    history: Tuple[Dict[str, Any], ...]
    train_time: timedelta


@serialize.instance(FitModelReturn)
def _serialize_fit_model_return(instance: FitModelReturn, path: Path) -> None:
    path = resolve(path, dir=True)
    for field_name, field in shallow_asdict(instance).items():
        save(field, path / field_name)


@deserialize.dispatch(FitModelReturn)
def _deserialize_fit_model_return(path: Path, _metadata: Metadata) -> FitModelReturn:
    return FitModelReturn(
        **{field.name: load(path / field.name) for field in fields(FitModelReturn)}
    )


def get_fit_model(
    compiled_model: CompiledModel,
    datasets: Described[DatasetTriplet[tf.data.Dataset], json.JSONDataType],
    params: FitModelParams,
    *,
    epoch_registry: RegisterEpoch,
    history_registry: SaveHistory,
) -> FitModelReturn:
    ds_train, ds_val, _ = datasets.value

    with Timer() as timer:
        compiled_model.architecture.model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=params.epochs,
            callbacks=params.callbacks.value + [history_registry, epoch_registry],
        )

    duration = timer.duration
    assert duration is not None

    return FitModelReturn(
        architecture=compiled_model.architecture,
        trained_epochs=epoch_registry.last_epoch(),
        history=tuple(history_registry.history),
        train_time=duration,
    )


@contextmanager
def strategy_scope(strategy: Optional[Described[tf.distribute.Strategy, Any]]) -> Iterator[None]:
    context = strategy.value.scope() if strategy is not None else nullcontext()

    with context:
        yield
