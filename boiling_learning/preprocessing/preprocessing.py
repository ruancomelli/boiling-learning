import os
from typing import Callable, Optional, TypeVar

import modin.pandas as pd
import scipy
import skimage
import tensorflow as tf
from dataclassy import dataclass
from tensorflow.data import AUTOTUNE

from boiling_learning.utils.utils import PathLike, resolve

T = TypeVar('T')


def interpolate_timeseries(x_ref, y_ref, x):
    f = scipy.interpolate.interp1d(x_ref, y_ref)
    return f(x)


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


@dataclass(frozen=True, kwargs=True)
class SizeSpec:
    height: int
    width: int


@dataclass(frozen=True, kwargs=True)
class CropSpec:
    offset_box: SizeSpec
    size: SizeSpec


def simple_image_preprocessor(
    interest_region: CropSpec, final_size: SizeSpec, downscale_factor: int
) -> Callable:
    def preprocessor(img):
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.crop_to_bounding_box(
            img,
            offset_height=interest_region.offset_box.height,
            offset_width=interest_region.offset_box.width,
            target_height=interest_region.size.height,
            target_width=interest_region.size.width,
        )
        img = tf.image.random_crop(img, (final_size.height, final_size.width, 1))
        img = skimage.transform.downscale_local_mean(img, (downscale_factor, downscale_factor, 1))

        return img

    return preprocessor


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
