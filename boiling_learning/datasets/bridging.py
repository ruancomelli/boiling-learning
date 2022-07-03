from __future__ import annotations

import math
from functools import partial
from typing import Callable, Hashable, List, Mapping, Optional, Tuple, TypeVar, Union

import tensorflow as tf

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.utils.functional import map_values
from boiling_learning.utils.mathutils import round_to_multiple
from boiling_learning.utils.utils import PathLike, resolve

_T = TypeVar('_T')
NestedStructure = Union[
    _T,
    List['NestedStructure[_T]'],
    Tuple['NestedStructure[_T]', ...],
    Mapping[Hashable, 'NestedStructure[_T]'],
]
NestedTypeSpec = NestedStructure[tf.TypeSpec]
NestedTensorLike = NestedStructure[tf.types.experimental.TensorLike]


def sliceable_dataset_to_tensorflow_dataset(
    dataset: SliceableDataset[_T],
    *,
    cache: bool = False,
    save_path: Optional[PathLike] = None,
    batch_size: Optional[int] = None,
    prefilterer: Optional[Callable[[_T], bool]] = None,
    filterer: Optional[Callable[..., bool]] = None,
    prefetch: int = 0,
    expand_to_batch_size: bool = False,
    deterministic: bool = False,
    target: Optional[str] = None,
) -> tf.data.Dataset:
    creator = partial(
        _create_tensorflow_dataset,
        dataset,
        prefilterer=prefilterer,
        prefetch=prefetch,
    )

    if save_path is None:
        ds = creator()
    else:
        save_path = resolve(save_path, parents=True)

        try:
            if not save_path.exists():
                raise FileNotFoundError

            ds = tf.data.experimental.load(str(save_path), auto_spec(dataset[0]))
        except FileNotFoundError:
            ds = creator()

            tf.data.experimental.save(ds, str(save_path))
            ds = tf.data.experimental.load(
                str(save_path),
                auto_spec(dataset[0]),
                reader_func=_make_reader_func(deterministic=deterministic),
            )

    if filterer is not None:
        ds = ds.filter(filterer)

    if target is not None:
        ds = ds.map(lambda feature, targets: (feature, targets[target]))

    if cache:
        ds = ds.cache()

    if batch_size is not None:
        if expand_to_batch_size:
            expanded_size = round_to_multiple(len(dataset), batch_size, rounder=math.ceil)
            ds = ds.repeat().take(expanded_size)

        ds = ds.batch(
            batch_size,
            drop_remainder=expand_to_batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            # deterministic=not shuffle,
            deterministic=deterministic,
        )

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def _make_reader_func(
    *, deterministic: bool = True
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    def _reader_func(datasets: tf.data.Dataset) -> tf.data.Dataset:
        return datasets.interleave(
            lambda dataset: dataset,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic,
        )

    return _reader_func


def _create_tensorflow_dataset(
    dataset: SliceableDataset[_T],
    *,
    prefilterer: Optional[Callable[[_T], bool]] = None,
    prefetch: int = 0,
) -> tf.data.Dataset:
    if prefetch:
        dataset = dataset.prefetch(prefetch)

    return tf.data.Dataset.from_generator(
        lambda: iter(dataset if prefilterer is None else filter(prefilterer, dataset)),
        output_signature=auto_spec(dataset[0]),
    )


def auto_spec(elem: NestedTensorLike) -> NestedTypeSpec:
    try:
        return tf.type_spec_from_value(elem)
    except TypeError:
        return map_values(auto_spec, elem)
