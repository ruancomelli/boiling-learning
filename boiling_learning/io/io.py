import collections
import io as _io
import json as _json
import pickle
import warnings
from contextlib import nullcontext
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type, TypeVar, Union

import cv2
import funcy
import h5py
import json_tricks
import numpy as np
import tensorflow as tf
import yogadl
import yogadl.storage
from tensorflow.keras.models import load_model

from boiling_learning.utils.dtypes import decode_element_spec, encode_element_spec
from boiling_learning.utils.functional import Kwargs
from boiling_learning.utils.utils import PathLike, ensure_dir, ensure_parent, is_, resolve

_T = TypeVar('_T')
_S = TypeVar('_S')
_Dataset = TypeVar('_Dataset')
SaverFunction = Callable[[_S, PathLike], Any]
LoaderFunction = Callable[[PathLike], _S]
DatasetTriplet = Tuple[_Dataset, Optional[_Dataset], _Dataset]
OptionalDatasetTriplet = Tuple[
    Optional[_Dataset],
    Optional[_Dataset],
    Optional[_Dataset],
]
BoolFlagged = Tuple[bool, _S]
BoolFlaggedLoaderFunction = LoaderFunction[BoolFlagged[_S]]


def add_bool_flag(
    loader: LoaderFunction[_T],
    expected_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = FileNotFoundError,
) -> BoolFlaggedLoaderFunction[Optional[_T]]:
    def _loader(path: PathLike) -> BoolFlagged[Optional[_T]]:
        try:
            return True, loader(path)
        except expected_exceptions:
            return False, None

    return _loader


def save_image(image: np.ndarray, path: PathLike) -> None:
    cv2.imwrite(str(ensure_parent(path)), image)


def load_image(path: PathLike, flag: Optional[int] = cv2.IMREAD_COLOR) -> np.ndarray:
    return cv2.imread(str(resolve(path)), flag)


def save_serialized(save_map: Mapping[_T, SaverFunction[_S]]) -> SaverFunction[Mapping[_T, _S]]:
    def save(return_dict: Mapping[_T, _S], path: PathLike) -> None:
        path = ensure_parent(path)
        for key, obj in return_dict.items():
            save_map[key](obj, path / key)

    return save


def load_serialized(load_map: Mapping[_T, LoaderFunction[_S]]) -> LoaderFunction[Dict[_T, _S]]:
    def load(path: PathLike) -> Dict[_T, _S]:
        path = resolve(path)
        return {key: loader(path / key) for key, loader in load_map.items()}

    return load


def save_keras_model(keras_model, path: PathLike, **kwargs) -> None:
    path = ensure_parent(path)
    keras_model.save(path, **kwargs)


def load_keras_model(path: PathLike, strategy=None, **kwargs):
    scope = strategy.scope() if strategy is not None else nullcontext()
    with scope:
        return load_model(path, **kwargs)


def save_pkl(obj, path: PathLike) -> None:
    path = ensure_parent(path)

    with path.open('wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path: PathLike):
    with resolve(path).open('rb') as file:
        return pickle.load(file)


def save_json(
    obj: _T,
    path: PathLike,
    dump: Callable[[_T, _io.TextIOWrapper], Any] = _json.dump,
    cls: Optional[Type] = None,
) -> None:
    path = ensure_parent(path)

    if path.suffix != '.json':
        warnings.warn(
            f'A JSON file is expected, but *path* ends with "{path.suffix}"',
            category=RuntimeWarning,
        )

    dump = Kwargs({'cls': cls}).omit('cls', is_(None)).partial(dump)
    with path.open('w', encoding='utf-8') as file:
        dump(obj, file, indent=4, ensure_ascii=False)


def load_json(
    path: PathLike,
    load: Callable[[_io.TextIOWrapper], _T] = _json.load,
    cls: Optional[Type] = None,
) -> _T:
    path = resolve(path)

    if path.suffix != '.json':
        warnings.warn(
            f'A JSON file is expected, but *path* ends with "{path.suffix}"',
            category=RuntimeWarning,
        )

    load = Kwargs({'cls': cls}).omit('cls', is_(None)).partial(load)
    with path.open('r', encoding='utf-8') as file:
        return load(file)


def saver_hdf5(key: str = '') -> SaverFunction[Any]:
    def save_hdf5(obj, path: PathLike) -> None:
        path = ensure_parent(path)
        with h5py.File(str(path), 'w') as hf:
            hf.create_dataset(key, data=obj)

    return save_hdf5


def loader_hdf5(key: str = '') -> LoaderFunction[Any]:
    def load_hdf5(path: PathLike):
        path = resolve(path)
        with h5py.File(str(path), 'r') as hf:
            return hf.get(key)

    return load_hdf5


def save_element_spec(element_spec: tf.TensorSpec, path: PathLike) -> None:
    encoded_element_spec = encode_element_spec(element_spec)
    save_json(encoded_element_spec, path, dump=json_tricks.dump)


def load_element_spec(path: PathLike) -> tf.TensorSpec:
    encoded_element_spec = load_json(path, load=json_tricks.load)
    return decode_element_spec(encoded_element_spec)


def save_dataset(dataset: tf.data.Dataset, path: PathLike) -> None:
    path = ensure_dir(path)
    dataset_path = path / 'dataset.tensorflow'
    element_spec_path = path / 'element_spec.json'

    save_element_spec(dataset.element_spec, element_spec_path)
    tf.data.experimental.save(dataset, str(dataset_path))


def load_dataset(path: PathLike) -> tf.data.Dataset:
    path = resolve(path)
    dataset_path = path / 'dataset.tensorflow'
    element_spec_path = path / 'element_spec.json'

    element_spec = load_element_spec(element_spec_path)

    def recurse_fix(elem_spec):
        if isinstance(elem_spec, list):
            return tuple(map(recurse_fix, elem_spec))
        elif isinstance(elem_spec, collections.OrderedDict):
            return dict(funcy.walk_values(recurse_fix, elem_spec))
        else:
            return elem_spec

    element_spec = recurse_fix(element_spec)

    return tf.data.experimental.load(str(dataset_path), element_spec)


def saver_dataset_triplet(
    saver: SaverFunction[tf.data.Dataset],
) -> SaverFunction[DatasetTriplet]:
    def _saver(ds: DatasetTriplet, path: PathLike) -> None:
        ds_train, ds_val, ds_test = ds

        path = ensure_dir(path)
        saver(ds_train, path / 'train')
        if ds_val is not None:
            saver(ds_val, path / 'val')
        saver(ds_test, path / 'test')

    return _saver


def loader_dataset_triplet(
    loader: LoaderFunction[Optional[tf.data.Dataset]],
) -> LoaderFunction[OptionalDatasetTriplet]:
    def _loader(path: PathLike) -> OptionalDatasetTriplet:
        path = resolve(path)

        ds_train = loader(path / 'train')
        ds_val = loader(path / 'val')
        ds_test = loader(path / 'test')

        return ds_train, ds_val, ds_test

    return _loader


def bool_flagged_loader_dataset_triplet(
    loader: BoolFlaggedLoaderFunction[Optional[tf.data.Dataset]],
) -> BoolFlaggedLoaderFunction[OptionalDatasetTriplet]:
    def _loader(path: PathLike) -> BoolFlagged[OptionalDatasetTriplet]:
        path = resolve(path)

        success_train, ds_train = loader(path / 'train')
        success_val, ds_val = loader(path / 'val')
        success_test, ds_test = loader(path / 'test')

        success = success_train and success_test
        if not success_val:
            ds_val = None

        return success, (ds_train, ds_val, ds_test)

    return _loader


def save_yogadl(
    dataset,
    storage_path: PathLike,
    dataset_id: str,
    dataset_version: str = '0.0',
) -> None:
    storage_path = ensure_dir(storage_path)

    lfs_config = yogadl.storage.LFSConfigurations(str(storage_path))
    storage = yogadl.storage.LFSStorage(lfs_config)
    storage.submit(dataset, dataset_id, dataset_version)


def saver_yogadl(storage_path: PathLike, dataset_id: str) -> SaverFunction[DatasetTriplet]:
    storage_path = resolve(storage_path)
    id_train = dataset_id + '_train'
    id_val = dataset_id + '_val'
    id_test = dataset_id + '_test'

    def _saver(ds: DatasetTriplet, path: Optional[PathLike] = None) -> None:
        ds_train, ds_val, ds_test = ds

        save_yogadl(ds_train, storage_path=storage_path, dataset_id=id_train)
        if ds_val is not None:
            save_yogadl(ds_val, storage_path=storage_path, dataset_id=id_val)
        save_yogadl(ds_test, storage_path=storage_path, dataset_id=id_test)

    return _saver


def load_yogadl(
    storage_path: PathLike,
    dataset_id: str,
    dataset_version: str = '0.0',
    start_offset: int = 0,
    shuffle: bool = False,
    skip_shuffle_at_epoch_end: bool = False,
    shuffle_seed: Optional[int] = None,
    shard_rank: int = 0,
    num_shards: int = 1,
    drop_shard_remainder: bool = False,
) -> tf.data.Dataset:
    storage_path = resolve(storage_path)

    lfs_config = yogadl.storage.LFSConfigurations(str(storage_path))
    storage = yogadl.storage.LFSStorage(lfs_config)
    dataref = storage.fetch(dataset_id, dataset_version)
    stream = dataref.stream(
        start_offset=start_offset,
        shuffle=shuffle,
        skip_shuffle_at_epoch_end=skip_shuffle_at_epoch_end,
        shuffle_seed=shuffle_seed,
        shard_rank=shard_rank,
        num_shards=num_shards,
        drop_shard_remainder=drop_shard_remainder,
    )
    return yogadl.tensorflow.make_tf_dataset(stream)


def loader_yogadl(storage_path: PathLike, dataset_id: str) -> LoaderFunction[DatasetTriplet]:
    storage_path = resolve(storage_path)
    id_train = dataset_id + '_train'
    id_val = dataset_id + '_val'
    id_test = dataset_id + '_test'

    def _loader(path: Optional[PathLike] = None):
        try:
            ds_train = load_yogadl(storage_path=storage_path, dataset_id=id_train)
        except AssertionError:
            ds_train = None

        try:
            ds_val = load_yogadl(storage_path=storage_path, dataset_id=id_val)
        except AssertionError:
            ds_val = None

        try:
            ds_test = load_yogadl(storage_path=storage_path, dataset_id=id_test)
        except AssertionError:
            ds_test = None

        success = ds_train is not None and ds_test is not None
        return success, (ds_train, ds_val, ds_test)

    return _loader
