import os
from typing import Callable, Optional, TypeVar

import modin.pandas as pd
import tensorflow as tf
from tensorflow.data import AUTOTUNE

from boiling_learning.utils.utils import PathLike, resolve

T = TypeVar('T')


def sync_dataframes(
    source_df: pd.DataFrame,
    dest_df: pd.DataFrame,
    source_time_column: Optional[str] = None,
    dest_time_column: Optional[str] = None,
) -> pd.DataFrame:
    allowed_index = (pd.DatetimeIndex, pd.TimedeltaIndex, pd.Float64Index)

    if source_time_column is not None:
        source_df = source_df.set_index(source_time_column, drop=False)
    if not isinstance(source_df.index, allowed_index):
        raise ValueError(
            f'the source DataFrame index must be one of {allowed_index}.'
            ' Ensure this or pass a valid column name as input.'
            f' Got {type(source_df.index)}'
        )

    if dest_time_column is not None:
        dest_df = dest_df.set_index(dest_time_column, drop=False)
    if not isinstance(dest_df.index, allowed_index):
        raise ValueError(
            f'the dest DataFrame index must be one of {allowed_index}.'
            ' Ensure this or pass a valid column name as input.'
            f' Got {type(dest_df.index)}'
        )

    if isinstance(source_df.index, pd.TimedeltaIndex):
        source_df.index = source_df.index.total_seconds()

    if isinstance(dest_df.index, pd.TimedeltaIndex):
        dest_df.index = dest_df.index.total_seconds()

    if type(source_df.index) is not type(dest_df.index):
        raise ValueError(
            f'the source and dest DataFrames indices must be the same type.'
            f' Got {type(source_df.index)} and {type(dest_df.index)}'
        )

    concat = pd.concat([source_df, dest_df]).sort_index()
    if isinstance(source_df.index, pd.Float64Index):
        concat = concat.interpolate(method='index', limit_direction='both')
    else:
        concat = concat.interpolate(method='time', limit_direction='both')
    concat = concat.loc[dest_df.index]
    return concat


def snapshotter(
    snapshot_folder: PathLike,
    num_shards: Optional[int] = None,
    shuffle_size: Optional[int] = None,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    snapshot_folder = resolve(snapshot_folder)

    if shuffle_size is None:
        shuffle_size = os.cpu_count()

    if num_shards is None:
        num_shards = shuffle_size

    def reader_fn(datasets: tf.data.Dataset) -> tf.data.Dataset:
        # shuffle the datasets splits
        datasets = datasets.shuffle(shuffle_size)
        # read datasets in parallel and interleave their elements
        return datasets.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)

    def op(ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = ds.enumerate()
        ds = ds.snapshot(
            str(snapshot_folder),
            reader_func=reader_fn,
            shard_func=lambda idx, value: idx % num_shards,
        )
        return ds.map(lambda idx, value: value)

    return op
