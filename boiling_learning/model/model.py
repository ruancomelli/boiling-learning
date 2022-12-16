from __future__ import annotations

import itertools
import json as _json
import operator
import typing
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, TypeVar

import more_itertools as mit
import tensorflow as tf
from pint import Quantity

from boiling_learning.io import json
from boiling_learning.io.storage import Metadata, deserialize, serialize
from boiling_learning.model.layers import ImageNormalization, RandomBrightness
from boiling_learning.utils.pathutils import resolve
from boiling_learning.utils.units import unit_registry as ureg

_CUSTOM_LAYERS = (ImageNormalization, RandomBrightness)
_CUSTOM_OBJECTS = {layer.__name__: layer for layer in _CUSTOM_LAYERS}

_Any = TypeVar('_Any')
Evaluation = dict[str, Any]


class ModelArchitecture:
    def __init__(self, model: tf.keras.models.Model) -> None:
        self.model = model

    @classmethod
    def from_inputs_and_outputs(
        cls, inputs: tf.keras.layers.Layer, outputs: tf.keras.layers.Layer
    ) -> ModelArchitecture:
        return cls(tf.keras.models.Model(inputs=inputs, outputs=outputs))

    def get_config(self) -> dict[str, Any]:
        return json.encode(self)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ModelArchitecture:
        return cls(
            tf.keras.models.model_from_json(_json.dumps(config), custom_objects=_CUSTOM_OBJECTS)
        )

    def __describe__(self) -> dict[str, Any]:
        model_json = _json.loads(self.model.to_json())
        return anonymize_model_json(
            {key: value for key, value in model_json['config'].items() if key != 'name'}
        )

    def count_parameters(self, *, trainable: bool = True, non_trainable: bool = True) -> int:
        if trainable and non_trainable:
            return typing.cast(int, self.model.count_params())
        elif trainable:
            return sum(tf.keras.backend.count_params(p) for p in self.model.trainable_weights)
        elif non_trainable:
            return sum(tf.keras.backend.count_params(p) for p in self.model.non_trainable_weights)
        else:
            raise ValueError('at least one of `trainable` and `non_trainable` must be true')

    def evaluate(self, data: tf.data.Dataset) -> Evaluation:
        return self.model.evaluate(data, return_dict=True)

    def clone(self) -> ModelArchitecture:
        return ModelArchitecture(tf.keras.models.clone_model(self.model))


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

    architecture = ModelArchitecture.from_config(_json.loads(model_json_path.read_text()))
    architecture.model.load_weights(str(model_weights_path))
    return architecture


def _model_memory_usage_in_bytes(
    architecture: ModelArchitecture, *, batch_size: int
) -> Quantity[int]:
    """Return the estimated memory usage of a given Keras model in bytes.

    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Source: https://stackoverflow.com/a/64359137/5811400

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.
    """
    model = architecture.model

    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count: Quantity[int] = 0 * ureg.byte
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += _model_memory_usage_in_bytes(layer, batch_size=batch_size)

        single_layer_mem: int = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    model_params_count = architecture.count_parameters()
    return (
        batch_size * shapes_mem_count + model_params_count
    ) * ureg.byte + internal_model_mem_count


def rename_model_layers(
    model: ModelArchitecture,
    renamer: Optional[Callable[[str], str]] = None,
    custom_objects: Any = _CUSTOM_OBJECTS,
) -> ModelArchitecture:
    '''Rename layers and model while keeping the pre-trained weights.

    Source: https://nrasadi.medium.com/change-model-layer-name-in-tensorflow-keras-58771dd6bf1b.
    '''
    if renamer is None:
        renamer = _default_renamer()

    config = model.model.get_config()
    old_to_new = {}
    new_to_old = {}

    for layer in config['layers']:
        new_name = renamer(layer['name'])
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                in_node[0] = old_to_new[in_node[0]]

    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]

    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]

    config['name'] = renamer(config['name'])
    new_model = tf.keras.Model.from_config(config, custom_objects)

    for layer in new_model.layers:
        layer.set_weights(model.model.get_layer(new_to_old[layer.name]).get_weights())

    return ModelArchitecture(new_model)


def anonymize_model_json(model_json: dict[str, Any]) -> dict[str, Any]:
    names = _collect_names(model_json, name_key='name')
    translator = {name: f'layer_{index}' for index, name in enumerate(mit.unique_everseen(names))}
    return _rename_layers(model_json, translator)


def _collect_names(config: Any, *, name_key: str = 'name') -> list[str]:
    names: list[str] = []
    if isinstance(config, dict):
        # sort keys and values here to ensure that they are always iterated over in the
        # same order for different models
        for key, value in sorted(config.items(), key=operator.itemgetter(0)):
            if key == name_key and isinstance(value, str):
                names.append(value)
            else:
                names.extend(_collect_names(value))
    elif isinstance(config, list):
        for value in config:
            names.extend(_collect_names(value))
    return names


def _rename_layers(config: _Any, translator: Mapping[str, str]) -> _Any:
    if isinstance(config, dict):
        return {key: _rename_layers(value, translator) for key, value in config.items()}
    elif isinstance(config, list):
        return [_rename_layers(value, translator) for value in config]
    else:
        return translator[config] if isinstance(config, str) and config in translator else config


def _default_renamer() -> Callable[[str], str]:
    counter = itertools.count()

    def _rename(_name: str) -> str:
        return f'layer_{next(counter)}'

    return _rename
