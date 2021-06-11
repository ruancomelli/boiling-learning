import itertools as it
import operator
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar

import funcy
import modin.pandas as pd
import more_itertools as mit
import scipy
import skimage
import tensorflow as tf
from dataclassy import dataclass
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from tensorflow.data.experimental import AUTOTUNE

import boiling_learning as bl
import boiling_learning.model as bl_model
import boiling_learning.utils as bl_utils
from boiling_learning.utils.utils import PathLike

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


def load_persistent(path, auto_purge: bool = False):
    def imread_as_float(path):
        try:
            return img_as_float(imread(path))
        except (SyntaxError, IOError):
            if auto_purge:
                path.unlink()
            raise

    def imsave_as_ubyte(path, img):
        return imsave(path, img_as_ubyte(img))

    return bl.management.Persistent(
        path,
        checker=operator.methodcaller('is_file'),
        reader=imread_as_float,
        writer=imsave_as_ubyte,
        record_paths=True,
    )


class TransformationPipeline:
    def __init__(self, *transformers):
        self._pipe = funcy.rcompose(*transformers)

    def transform(
        self,
        X,
        many=False,
        fetch=None,
        # parallel=False
    ):
        should_fetch = callable(fetch) or bool(fetch)

        if many:
            transformer = partial(self.transform, many=False, fetch=fetch)

            # if parallel:
            # from pathos.multiprocessing import ProcessingPool
            # # from multiprocessing import Pool

            # if isinstance(parallel, bool):
            #     processes = []
            # else:
            #     processes = [parallel]

            # with ProcessingPool(*processes) as pool:
            # # with Pool(*processes) as pool:
            #     bl_utils.print_header(f'Creating pool with ncpus={pool.ncpus} and nodes={pool.nodes}')
            #     if should_fetch:
            #         return pool.map(transformer, X)
            #         # return pool.map(bl_utils.worker.apply_to_f, ((transformer, x, {}) for x in X))
            #         # return pool.map(bl_utils.worker.apply_to_obj, ((self, 'transform', x, {}) for x in X))
            #     else:
            #         mit.consume(pool.uimap(transformer, X))
            #         # mit.consume(pool.imap_unordered(transformer, X))
            #         # mit.consume(pool.imap_unordered(bl_utils.worker.apply_to_f, ((transformer, x, {}) for x in X)))
            #         # mit.consume(pool.imap_unordered(bl_utils.worker.apply_to_obj, ((self, 'transform', x, {}) for x in X)))
            #         return self
            # else:
            mapper = map(transformer, X)
            if should_fetch:
                return list(mapper)
            else:
                mit.consume(mapper)
                return self
        else:
            if should_fetch:
                if callable(fetch):
                    return fetch(self._pipe(X))
                else:
                    return self._pipe(X)
            else:
                self._pipe(X)
                return self


class ImageDatasetTransformer:
    def __init__(
        self,
        transformers,
        loader=None,
        saver=None,
        persist_intermediate=False,
        persist_last=True,
        auto_purge=False,
    ):
        # transformers is an iterable yielding (path_transformer, value_transformer)

        if loader is None:
            loader = partial(load_persistent, auto_purge=auto_purge)
        self.loader = loader

        if saver is None:
            saver = bl.management.Persistent.persist
        self.saver = saver

        self.transformers = transformers
        self._persist_intermediate = persist_intermediate
        self._persist_last = persist_last
        self._pipe = None
        self._assembled = False

    def _assemble(self):
        if self._persist_intermediate:
            transformers = mit.intersperse(self.saver, self.transformers)
        else:
            transformers = self.transformers

        transformers = mit.prepend(self.loader, transformers)

        if self._persist_last:
            transformers = bl_utils.append(
                transformers, bl.management.Persistent.persist
            )

        self._assembled = True
        self._pipe = TransformationPipeline(*transformers)

    def transform_images(self, images, **kwargs):
        if not self._assembled:
            self._assemble()
        return self._pipe.transform(images, many=True, **kwargs)


# TODO: test this


class DatasetTransformerTF(bl_utils.SimpleRepr, bl_utils.SimpleStr):
    def __init__(
        self,
        path_transformers,
        transformers,
        loader: Callable[[PathLike], Any],
        saver: Optional[Callable] = None,
        save_intermediate: bool = True,
        save_last: bool = True,
        batch_size: Optional[int] = None,
    ):
        self.path_transformers = path_transformers
        self.transformers = transformers
        self.loader = loader
        self.saver = saver
        self.save_intermediate: bool = save_intermediate
        self.batch_size: Optional[int] = batch_size

    def _load_tensor(self, sources):
        ds = tf.data.Dataset.from_generator(
            lambda: map(str, sources), tf.string
        )
        ds = ds.map(self.loader, num_parallel_calls=AUTOTUNE)
        ds = ds.prefetch(AUTOTUNE)

        return ds

    def _save_tensor(self, dests, ds):
        if self.batch_size is None:
            for dest, img in zip(dests, ds.as_numpy_iterator()):
                self.saver(img, dest)
        else:
            for dest_chunk, img_chunk in zip(
                mit.ichunked(dests, self.batch_size),
                ds.batch(self.batch_size).as_numpy_iterator(),
            ):
                for dest, img in zip(dest_chunk, img_chunk):
                    self.saver(img, dest)

    def _full_trajectories(self, sources: Iterable[PathLike]):
        sources = map(Path, sources)

        def trajectory(source):
            return it.accumulate(
                # self.path_transformers, # Python 3.8 only
                mit.prepend(source, self.path_transformers),
                lambda current_path, path_transformer: path_transformer(
                    current_path
                ),
                # initial=source # Python 3.8 only
            )

        trajs = map(trajectory, sources)
        trajs = map(list, trajs)

        return trajs

    def _valid_trajectories(
        self, trajs, erased_marker, cmp_marker=operator.is_
    ):
        def split_source_dest(traj):
            traj, erased = mit.partition(
                partial(cmp_marker, erased_marker), traj
            )
            dests, possible_sources = mit.partition(
                operator.methodcaller('is_file'), traj
            )

            possible_sources = list(possible_sources)
            source = possible_sources.pop()

            erased = it.chain(
                erased, it.repeat(erased_marker, len(possible_sources))
            )

            return list(erased) + [source] + list(dests)

        trajs = map(split_source_dest, trajs)

        return trajs

    def _step_from_idx(
        self, trajs, step_idx, erased_marker, cmp_marker=operator.is_
    ):
        trajs = it.filterfalse(
            # removes erased trajectories, i.e., the ones that already exist and don't need to be transformed
            # lambda traj: cmp_marker(traj[step_idx], erased_marker),
            funcy.compose(
                partial(cmp_marker, erased_marker),
                operator.itemgetter(step_idx),
            ),
            trajs,
        )
        trajs = map(operator.itemgetter(step_idx, step_idx + 1), trajs)
        trajs = mit.peekable(trajs)

        if trajs:
            sources, dests = mit.unzip(trajs)
        else:
            sources = bl_utils.empty_gen()
            dests = bl_utils.empty_gen()

        return sources, dests

    def transform_paths(self, paths: Iterable[PathLike]):
        return map(funcy.rcompose(*self.path_transformers), paths)

    def transform_dataset(self, ds):
        return ds.map(
            funcy.rcompose(*self.transformers), num_parallel_calls=AUTOTUNE
        )

    def _transform_images_indirect(self, paths: Iterable[PathLike]):
        erased_marker = None
        full_trajs = tuple(self._full_trajectories(paths))

        trajs = full_trajs
        for step_idx, transformer in enumerate(self.transformers):
            trajs = self._valid_trajectories(
                trajs, erased_marker=erased_marker
            )
            trajs = tuple(trajs)
            sources, dests = self._step_from_idx(
                trajs, step_idx, erased_marker=erased_marker
            )

            ds = self._load_tensor(sources)
            ds = ds.map(transformer, num_parallel_calls=AUTOTUNE)
            self._save_tensor(dests, ds)

        final = map(mit.last, full_trajs)
        ds = self._load_tensor(final)

        return full_trajs, ds

    def _transform_images_direct(self, paths: Iterable[PathLike]):
        paths = tuple(paths)
        dests = tuple(self.transform_paths(paths))

        ds = self._load_tensor(paths)
        ds = self.transform_dataset(ds)

        if self.save_last:
            self._save_tensor(dests, ds)

        full_trajs = tuple(zip(paths, dests))
        return full_trajs, ds

    def transform_images(self, paths: Iterable[PathLike]):
        if self.save_intermediate:
            return self._transform_images_indirect(paths)
        else:
            return self._transform_images_direct(paths)


class ImageDatasetTransformerTF(bl_utils.SimpleRepr, bl_utils.SimpleStr):
    '''Transforms a sequence of images using a sequence of transformations.'''

    def __init__(
        self,
        path_transformers,
        transformers,
        batch_size,
        loader,
        saver,
        split_id='all',
        chunk_index: Optional[int] = None,
        chunk_size: Optional[int] = None,
        n_chunks: Optional[int] = None,
    ):
        self.path_transformers = path_transformers
        self.transformers = transformers
        self.batch_size = batch_size
        self.loader = loader
        self.saver = saver

        self.split_id = bl_model.Split.get_split(split_id)

        if (chunk_index is None) ^ (
            (chunk_size is None) ^ (n_chunks is not None)
        ):
            raise ValueError(
                'chunk_index must be passed with either chunk_size or n_chunks, or they all must be omitted.'
            )

        if (chunk_size is not None) and (n_chunks is not None):
            raise ValueError(
                'either chunk_size or n_chunks (or both) must be None.'
            )

        self.chunk_size: Optional[int] = chunk_size
        self.chunk_index: Optional[int] = chunk_index
        self.chunk_index: Optional[int] = chunk_index

    def is_using_chunks(self):
        return self.chunk_size is not None

    def _extract_paths(self, img_ds, split_id=None):
        if split_id is bl_model.Split.TRAIN:
            df = img_ds.train_paths
        elif split_id is bl_model.Split.VAL:
            df = img_ds.val_paths
        elif split_id is bl_model.Split.TEST:
            df = img_ds.test_paths
        elif split_id is bl_model.Split.ALL:
            df = img_ds.paths
        else:
            raise ValueError(f'split_id={split_id} not supported.')
        df = self._get_chunk(df)

        return df

    def _get_chunk(self, iterable: Iterable[T]) -> Iterable[T]:
        if self.chunk_size is not None:
            chunks = mit.ichunked(iterable, self.chunk_size)
            chunk = mit.nth_or_last(chunks, self.chunk_index)
            return chunk
        elif self.n_chunks is not None:
            iterable = tuple(iterable)
            n = len(iterable)
            idx = self.chunk_index
            chunk = iterable[idx * n : idx * (n + 1)]

            return chunk
        else:
            return iterable

    def _load_tensor(self, sources):
        sources = map(str, sources)

        ds = tf.data.Dataset.from_generator(lambda: sources, tf.string)
        ds = ds.map(self.loader, num_parallel_calls=AUTOTUNE)
        ds = ds.prefetch(AUTOTUNE)

        return ds

    def _save_tensor(self, dests, ds):
        for dest_chunk, img_chunk in zip(
            mit.ichunked(dests, self.batch_size),
            ds.batch(self.batch_size).as_numpy_iterator(),
        ):
            for dest, img in zip(dest_chunk, img_chunk):
                self.saver(img, dest)

    def _full_trajectories(self, img_ds):
        sources = self._extract_paths(img_ds, self.split_id)
        sources = map(Path, sources)

        def trajectory(source):
            return it.accumulate(
                # self.path_transformers, # Python 3.8 only
                mit.prepend(source, self.path_transformers),
                lambda current_path, path_transformer: path_transformer(
                    current_path
                ),
                # initial=source # Python 3.8 only
            )

        trajs = map(trajectory, sources)
        trajs = map(list, trajs)

        return trajs

    def _valid_trajectories(
        self, trajs, erased_marker, cmp_marker=operator.is_
    ):
        def split_source_dest(traj):
            traj, erased = mit.partition(
                partial(cmp_marker, erased_marker), traj
            )
            dests, possible_sources = mit.partition(
                operator.methodcaller('is_file'), traj
            )

            possible_sources = list(possible_sources)
            source = possible_sources.pop()

            erased = it.chain(
                erased, it.repeat(erased_marker, len(possible_sources))
            )

            return list(erased) + [source] + list(dests)

        trajs = map(split_source_dest, trajs)

        return trajs

    def _step_from_idx(
        self, trajs, step_idx, erased_marker, cmp_marker=operator.is_
    ):
        trajs = it.filterfalse(
            # removes erased trajectories, i.e., the ones that already exist and don't need to be transformed
            lambda traj: cmp_marker(traj[step_idx], erased_marker),
            trajs,
        )
        trajs = map(operator.itemgetter(step_idx, step_idx + 1), trajs)
        trajs = mit.peekable(trajs)

        if trajs:
            sources, dests = mit.unzip(trajs)
        else:
            sources = bl_utils.empty_gen()
            dests = bl_utils.empty_gen()

        return sources, dests

    def transform_images(self, img_ds):
        erased_marker = None
        full_trajs = list(self._full_trajectories(img_ds))

        trajs = full_trajs
        for step_idx, transformer in enumerate(self.transformers):
            trajs = self._valid_trajectories(
                trajs, erased_marker=erased_marker
            )
            trajs = list(trajs)
            sources, dests = self._step_from_idx(
                trajs, step_idx, erased_marker=erased_marker
            )

            ds = self._load_tensor(sources)
            ds = ds.map(transformer, num_parallel_calls=AUTOTUNE)

            self._save_tensor(dests, ds)

        final = map(mit.last, full_trajs)
        ds = self._load_tensor(final)

        return full_trajs, ds


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
        img = tf.image.random_crop(
            img, (final_size.height, final_size.width, 1)
        )
        img = skimage.transform.downscale_local_mean(
            img, (downscale_factor, downscale_factor, 1)
        )

        return img

    return preprocessor


def snapshotter(
    snapshot_folder: PathLike,
    num_shards: Optional[int] = None,
    shuffle_size: Optional[int] = None,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    snapshot_folder = bl_utils.ensure_resolved(snapshot_folder)

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
        ds = ds.apply(
            tf.data.experimental.snapshot(
                str(snapshot_folder),
                reader_func=reader_fn,
                shard_func=lambda idx, value: idx % num_shards,
            )
        )
        return ds.map(lambda idx, value: value)

    return op
