from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Union

from dataclassy import dataclass
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer

from boiling_learning.io import json
from boiling_learning.utils.descriptions import describe
from boiling_learning.utils.typeutils import typename
from boiling_learning.utils.utils import JSONDataType


@describe.instance(Metric)
def _describe_configurable(instance: Metric) -> str:
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
    return CompiledModel()
