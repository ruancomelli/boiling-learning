import io as _io
import json as _json
import warnings
from contextlib import nullcontext
from typing import Any, Callable, Generic, Optional, Tuple, Type, TypeVar

import json_tricks
import tensorflow as tf
from tensorflow.keras.models import load_model

from boiling_learning.utils.dtypes import decode_element_spec, encode_element_spec
from boiling_learning.utils.functional import Kwargs
from boiling_learning.utils.utils import PathLike, is_, resolve

_T = TypeVar('_T')
SaverFunction = Callable[[_T, PathLike], Any]
LoaderFunction = Callable[[PathLike], _T]
BoolFlagged = Tuple[bool, _T]
BoolFlaggedLoaderFunction = LoaderFunction[BoolFlagged[_T]]


class DatasetTriplet(Tuple[_T, Optional[_T], _T], Generic[_T]):
    pass


# TODO: replicate the `strategy` behavior in the new model loading functionality
def load_keras_model(path: PathLike, strategy=None, **kwargs):
    scope = strategy.scope() if strategy is not None else nullcontext()
    with scope:
        return load_model(path, **kwargs)


def save_json(
    obj: _T,
    path: PathLike,
    dump: Callable[[_T, _io.TextIOWrapper], Any] = _json.dump,
    cls: Optional[Type] = None,
) -> None:
    path = resolve(path, parents=True)

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


def save_element_spec(element_spec: tf.TensorSpec, path: PathLike) -> None:
    encoded_element_spec = encode_element_spec(element_spec)
    save_json(encoded_element_spec, path, dump=json_tricks.dump)


def load_element_spec(path: PathLike) -> tf.TensorSpec:
    encoded_element_spec = load_json(path, load=json_tricks.load)
    return decode_element_spec(encoded_element_spec)
