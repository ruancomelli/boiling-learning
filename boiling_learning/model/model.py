from __future__ import annotations

import enum
import json as _json
import typing
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf

from boiling_learning.io import json
from boiling_learning.io.storage import Metadata, deserialize, serialize
from boiling_learning.utils.dataclasses import dataclass
from boiling_learning.utils.utils import resolve


@dataclass(frozen=True)
class ModelArchitecture:
    model: tf.keras.models.Model

    @classmethod
    def from_inputs_and_outputs(
        cls, inputs: tf.keras.layers.Layer, outputs: tf.keras.layers.Layer
    ) -> ModelArchitecture:
        return cls(tf.keras.models.Model(inputs=inputs, outputs=outputs))

    def __json_encode__(self) -> Dict[str, Any]:
        return _anonymize_model(_json.loads(self.model.to_json()))

    def __describe__(self) -> Dict[str, Any]:
        return typing.cast(Dict[str, Any], json.encode(self))


@serialize.instance(ModelArchitecture)
def _serialize_model(instance: ModelArchitecture, path: Path) -> None:
    path = resolve(path, dir=True)
    model_json_path = path / 'model.json'
    model_weights_path = path / 'weights.h5'

    instance.model.save_weights(str(model_weights_path))
    model_json_path.write_text(instance.model.to_json())


@deserialize.dispatch(ModelArchitecture)
def _deserialize_model(path: Path, _metadata: Metadata) -> ModelArchitecture:
    path = resolve(path)
    model_json_path = path / 'model.json'
    model_weights_path = path / 'weights.h5'

    if not model_json_path.is_file():
        raise FileNotFoundError(str(model_json_path))

    if not model_weights_path.is_file():
        raise FileNotFoundError(str(model_weights_path))

    architecture = ModelArchitecture(tf.keras.models.model_from_json(model_json_path.read_text()))
    architecture.model.load_weights(str(model_weights_path))
    return architecture


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
