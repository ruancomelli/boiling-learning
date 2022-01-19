from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Union

from dataclassy import dataclass
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer

from boiling_learning.datasets.sliceable import (
    SupervisedSliceableDataset,
    sliceable_dataset_to_tensorflow_dataset,
)
from boiling_learning.io import json
from boiling_learning.io.io import DatasetTriplet
from boiling_learning.utils.described import Described
from boiling_learning.utils.descriptions import describe
from boiling_learning.utils.typeutils import typename
from boiling_learning.utils.utils import JSONDataType


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


@describe.instance(Model)
def _describe_model(instance: Model) -> JSONDataType:
    return json.loads(instance.to_json())


@dataclass(frozen=True)
class ModelArchitecture:
    model: Model


@dataclass(frozen=True)
class CompileModelParams:
    architecture: ModelArchitecture
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
    callbacks: Described[List[Callback], JSONDataType]


@dataclass(frozen=True)
class FitModel:
    model: Model
    datasets: Described[DatasetTriplet[SupervisedSliceableDataset], JSONDataType]
    compile_params: CompileModelParams
    fit_params: FitModelParams


def fit_model(
    compiled_model: CompiledModel,
    datasets: Described[DatasetTriplet[SupervisedSliceableDataset], JSONDataType],
    params: FitModelParams,
) -> FitModel:
    model = compiled_model.model
    model.fit(
        sliceable_dataset_to_tensorflow_dataset(datasets.value),
        batch_size=params.batch_size,
        epochs=params.epochs,
        callbacks=params.callbacks.value,
        use_multiprocessing=True,
        workers=-1,
    )
    return FitModel(model, datasets, compile_params=compiled_model.params, fit_params=params)
