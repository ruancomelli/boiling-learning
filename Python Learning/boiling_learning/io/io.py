import json
import pickle
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    TypeVar
)
import warnings

import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model

import boiling_learning.utils as bl_utils
from boiling_learning.utils import (
    PathType,
    nullcontext,
    ensure_dir,
    ensure_parent,
    ensure_resolved,
    dtypes
)

T = TypeVar('T')
S = TypeVar('S')
SaverFunction = Callable[[S, PathType], Any]
LoaderFunction = Callable[[PathType], S]


def save_serialized(
        save_map: Mapping[T, SaverFunction[S]]
) -> SaverFunction[Mapping[T, S]]:
    def save(return_dict: Mapping[T, S], path: PathType) -> None:
        path = ensure_parent(path)
        for key, obj in return_dict.items():
            save_map[key](obj, path / key)
    return save


def load_serialized(
        load_map: Mapping[T, LoaderFunction[S]]
) -> LoaderFunction[Dict[T, S]]:
    def load(path: PathType) -> Dict[T, S]:
        path = ensure_resolved(path)
        loaded = {
            key: loader(path / key)
            for key, loader in load_map.items()
        }
        return loaded
    return load


def save_keras_model(keras_model, path: PathType, **kwargs) -> None:
    path = ensure_parent(path)
    keras_model.save(path, **kwargs)


def load_keras_model(path: PathType, strategy=None, **kwargs):
    if strategy is None:
        scope = nullcontext()
    else:
        scope = strategy.scope()

    with scope:
        return load_model(path, **kwargs)


def save_pkl(obj, path: PathType) -> None:
    path = ensure_parent(path)

    with path.open('wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path: PathType):
    with ensure_resolved(path).open('rb') as file:
        return pickle.load(file)


def save_json(obj, path: PathType) -> None:
    path = ensure_parent(path)

    if path.suffix != '.json':
        warnings.warn(
            f'A JSON file is expected, but *path* ends with "{path.suffix}"',
            category=warnings.RuntimeWarning
        )

    with path.open('w', encoding='utf-8') as file:
        json.dump(obj, file, indent=4, ensure_ascii=False)


def load_json(path: PathType):
    path = ensure_resolved(path)

    if path.suffix != '.json':
        warnings.warn(
            f'A JSON file is expected, but *path* ends with "{path.suffix}"',
            category=warnings.RuntimeWarning
        )

    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def saver_hdf5(key: str = '') -> SaverFunction[Any]:
    def save_hdf5(obj, path: PathType) -> None:
        path = ensure_parent(path)
        with h5py.File(str(path), 'w') as hf:
            hf.create_dataset(key, data=obj)
    return save_hdf5


def loader_hdf5(key: str = '') -> LoaderFunction[Any]:
    def load_hdf5(path: PathType):
        path = ensure_resolved(path)
        with h5py.File(str(path), 'r') as hf:
            return hf.get(key)
    return load_hdf5


def _element_spec_to_list(element_spec):
    return [
        {
            'dtype': bl_utils.dtypes.tf_str_dtype_bidict.inverse[type_spec.shape.dtype],
            'shape': list(type_spec.shape)
        }
        for type_spec in element_spec
    ]


def _element_list_to_spec(lst):
    return tuple(
        tf.TensorSpec(
            shape=tuple(dct['shape']),
            dtype=bl_utils.dtypes.tf_str_dtype_bidict[dct['dtype']]
        )
        for dct in lst
    )


def save_dataset(dataset, path: PathType):
    path = ensure_dir(path)
    dataset_path = path / 'dataset'
    element_spec_path = path / 'element_spec.json'

    save_json(
        _element_spec_to_list(dataset.element_spec),
        element_spec_path
    )

    tf.data.experimental.save(dataset, dataset_path)


def load_dataset(path: PathType):
    path = ensure_resolved(path)
    dataset_path = path / 'dataset'
    element_spec_path = path / 'element_spec.json'

    element_spec_list = load_json(element_spec_path)
    element_spec = _element_list_to_spec(element_spec_list)

    return tf.data.experimental.load(dataset_path, element_spec)
