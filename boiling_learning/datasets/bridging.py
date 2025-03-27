from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from functools import partial
from typing import TypeVar

import funcy
import tensorflow as tf
from loguru import logger

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.utils.pathutils import PathLike, resolve

_T = TypeVar("_T")
NestedStructure = (
    _T
    | list["NestedStructure[_T]"]
    | tuple["NestedStructure[_T]", ...]
    | Mapping[Hashable, "NestedStructure[_T]"]
)
NestedTypeSpec = NestedStructure[tf.TypeSpec]
NestedTensorLike = NestedStructure[tf.types.experimental.TensorLike]


def sliceable_dataset_to_tensorflow_dataset(
    dataset: SliceableDataset[_T],
    *,
    cache: bool = False,
    save_path: PathLike | None = None,
    batch_size: int | None = None,
    prefilterer: Callable[[_T], bool] | None = None,
    filterer: Callable[..., bool] | None = None,
    prefetch: int = 0,
    deterministic: bool = False,
    target: str | None = None,
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

            logger.debug(f"Loading dataset from {save_path}")
            # TODO: now passing the typespec is optional... try removing it!
            ds = tf.data.Dataset.load(
                str(save_path),
                reader_func=_make_reader_func(deterministic=deterministic),
            )
        except FileNotFoundError:
            logger.debug(f"File does not exist: {save_path}")

            ds = creator()

            ds.save(str(save_path))
            ds = tf.data.Dataset.load(
                str(save_path),
                reader_func=_make_reader_func(deterministic=deterministic),
            )

    if filterer is not None:
        ds = ds.filter(filterer)

    if target is not None:
        ds = ds.map(lambda feature, targets: (feature, targets[target]))

    if cache:
        ds = ds.cache()

    if batch_size is not None:
        ds = ds.batch(
            batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
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
    prefilterer: Callable[[_T], bool] | None = None,
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
        walker = funcy.walk_values if hasattr(elem, "items") else funcy.walk  # type: ignore
        return walker(auto_spec, elem)
