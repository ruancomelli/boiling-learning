from __future__ import annotations

import enum
import json as _json
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf

from boiling_learning.io import json
from boiling_learning.io.storage import Metadata, deserialize, serialize
from boiling_learning.utils.dataclasses import dataclass
from boiling_learning.utils.descriptions import describe
from boiling_learning.utils.utils import resolve


class Model(tf.keras.models.Model):
    pass


@serialize.instance(Model)
def _serialize_model(instance: Model, path: Path) -> None:
    tf.keras.models.save_model(instance, resolve(path))


@deserialize.dispatch(Model)
def _deserialize_model(path: Path, _metadata: Metadata) -> Model:
    return tf.keras.models.load_model(path)


@json.encode.instance(Model)
@describe.instance(Model)
def _describe_model(instance: Model) -> json.JSONDataType:
    return _anonymize_model(_json.loads(instance.to_json()))


@dataclass(frozen=True)
class ModelArchitecture:
    model: Model


class ProblemType(enum.Enum):
    CLASSIFICATION = enum.auto()
    REGRESSION = enum.auto()


def _anonymize_model(model_json: Dict[str, Any]) -> Dict[str, Any]:
    model_config = model_json['config']

    # remove model name
    del model_config['name']

    layer_indices = {layer['name']: index for index, layer in enumerate(model_config['layers'])}

    json_str = _json.dumps(model_json)

    for name, index in layer_indices.items():
        json_str = json_str.replace(name, f'layer_{index}')

    return _json.loads(json_str)
