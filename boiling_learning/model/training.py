from __future__ import annotations

import json as _json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import tensorflow as tf
from dataclassy import dataclass
from loguru import logger
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer
from typing_extensions import TypedDict

from boiling_learning.datasets.sliceable import (
    SupervisedSliceableDataset,
    sliceable_dataset_to_tensorflow_dataset,
)
from boiling_learning.io import json
from boiling_learning.io.io import DatasetTriplet
from boiling_learning.model.model import Model
from boiling_learning.utils.described import Described
from boiling_learning.utils.descriptions import describe
from boiling_learning.utils.typeutils import typename


@describe.instance(Metric)
def _describe_typename(instance: Metric) -> str:
    return typename(instance)


class TypeAndConfig(TypedDict):
    type: str
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


@json.encode.instance(Model)
@describe.instance(Model)
def _describe_model(instance: Model) -> json.JSONDataType:
    return _anonymize_model(_json.loads(instance.to_json()))


@dataclass(frozen=True)
class ModelArchitecture:
    model: Model


@dataclass(frozen=True)
class CompileModelParams:
    loss: Loss
    optimizer: Optimizer
    metrics: Optional[List[Metric]]


@dataclass(frozen=True)
class CompiledModel:
    model: Model
    params: CompileModelParams


def get_compiled_model(architecture: ModelArchitecture, params: CompileModelParams) -> Model:
    model = architecture.model

    model.compile(optimizer=params.optimizer, loss=params.loss, metrics=params.metrics)

    return model


def compile_model(architecture: ModelArchitecture, params: CompileModelParams) -> CompiledModel:
    return CompiledModel(get_compiled_model(architecture, params), params)


@dataclass(frozen=True)
class FitModelParams:
    batch_size: Optional[int]
    epochs: int
    callbacks: Described[List[Callback], json.JSONDataType]


@dataclass(frozen=True)
class FitModel:
    model: Model
    datasets: Described[DatasetTriplet[SupervisedSliceableDataset], json.JSONDataType]
    compile_params: CompileModelParams
    fit_params: FitModelParams


def get_fit_model(
    compiled_model: CompiledModel,
    datasets: Described[DatasetTriplet[SupervisedSliceableDataset], json.JSONDataType],
    params: FitModelParams,
    *,
    cache: Optional[Path] = None,
) -> Model:
    model = compiled_model.model

    ds_train, ds_val, _ = datasets.value
    ds_train = sliceable_dataset_to_tensorflow_dataset(
        ds_train, batch_size=params.batch_size, shuffle=True, prefetch=True, cache=cache
    )
    ds_val = sliceable_dataset_to_tensorflow_dataset(
        ds_val, batch_size=params.batch_size, shuffle=True, prefetch=True, cache=cache
    )

    model.fit(
        ds_train,
        batch_size=params.batch_size,
        epochs=params.epochs,
        validation_data=ds_val,
        callbacks=params.callbacks.value,
        use_multiprocessing=True,
        workers=-1,
    )

    return model


def fit_model(
    compiled_model: CompiledModel,
    datasets: Described[DatasetTriplet[SupervisedSliceableDataset], json.JSONDataType],
    params: FitModelParams,
) -> FitModel:
    return FitModel(
        get_fit_model(compiled_model=compiled_model, datasets=datasets, params=params),
        datasets,
        compile_params=compiled_model.params,
        fit_params=params,
    )


@contextmanager
def strategy_scope(strategy: Optional[Described[tf.distribute.Strategy, Any]]) -> Iterator[None]:
    context = strategy.value.scope() if strategy is not None else nullcontext()

    with context:
        yield


def _anonymize_model(model_json: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug(model_json)

    model_config = model_json['config']

    # remove model name
    del model_config['name']

    layer_indices = {layer['name']: index for index, layer in enumerate(model_config['layers'])}

    json_str = _json.dumps(model_json)

    for name, index in layer_indices.items():
        json_str = json_str.replace(name, f'layer_{index}')

    return _json.loads(json_str)
