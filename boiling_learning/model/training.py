from __future__ import annotations

import json as _json
from typing import Any, Dict, List, Optional, Union

from dataclassy import dataclass
from tensorflow.keras import Model
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
    return _json.loads(instance.to_json())


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


def compile_model(architecture: ModelArchitecture, params: CompileModelParams) -> CompiledModel:
    model = architecture.model
    model.compile(optimizer=params.optimizer, loss=params.loss, metrics=params.metrics)
    return CompiledModel(model, params)


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


def fit_model(
    compiled_model: CompiledModel,
    datasets: Described[DatasetTriplet[SupervisedSliceableDataset], json.JSONDataType],
    params: FitModelParams,
) -> FitModel:
    model = compiled_model.model

    ds = datasets.value
    ds_train = sliceable_dataset_to_tensorflow_dataset(ds[0])
    ds_val = sliceable_dataset_to_tensorflow_dataset(ds[1])

    if params.batch_size is not None:
        ds_train = ds_train.batch(params.batch_size)
        ds_val = ds_val.batch(params.batch_size)

    model.fit(
        ds_train,
        batch_size=params.batch_size,
        epochs=params.epochs,
        validation_data=ds_val,
        callbacks=params.callbacks.value,
        use_multiprocessing=True,
        workers=-1,
    )
    return FitModel(model, datasets, compile_params=compiled_model.params, fit_params=params)
