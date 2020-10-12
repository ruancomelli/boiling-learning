import collections
import itertools
import operator
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Mapping, Optional,
                    Tuple, Type, Union, overload)

import modin.pandas as pd
import numpy as np
import tensorflow as tf

import boiling_learning.utils as bl_utils
from boiling_learning.utils import PathType, VerboseType
import boiling_learning.io as bl_io
from boiling_learning.preprocessing.ExperimentVideo import ExperimentVideo


@bl_utils.simple_pprint_class
class ImageDataset(typing.MutableMapping[str, ExperimentVideo]):
    '''
    TODO: improve this
    An ImageDataset is a file CSV in df_path and the correspondent images. The file in df_path contains at least two columns. One of this columns contains file paths, and the other the targets for training, validation or test. This is intended for using flow_from_dataframe. There may be an optional column which specifies if that image belongs to the training, the validation or the test sets.
    '''

    VideoData: Type[ExperimentVideo.VideoData] = ExperimentVideo.VideoData
    DataFrameColumnNames: Type[ExperimentVideo.DataFrameColumnNames] = ExperimentVideo.DataFrameColumnNames
    DataFrameColumnTypes: Type[ExperimentVideo.DataFrameColumnTypes] = ExperimentVideo.DataFrameColumnTypes

    @dataclass(frozen=True)
    class VideoDataKeys(ExperimentVideo.VideoDataKeys):
        name: str = 'name'
        ignore: str = 'ignore'

    def __init__(
            self,
            name: str,
            column_names: DataFrameColumnNames = DataFrameColumnNames(),
            column_types: DataFrameColumnTypes = DataFrameColumnTypes(),
            df_path: Optional[PathType] = None,
            exist_load: bool = False
    ):
        self._name: str = name
        self.column_names: self.DataFrameColumnNames = column_names
        self.column_types: self.DataFrameColumnTypes = column_types
        self._experiment_videos: Dict[str, ExperimentVideo] = {}
        self._allow_key_overwrite: bool = True
        self.df: Optional[pd.DataFrame] = None
        self.ds = None

        if df_path is not None:
            df_path = bl_utils.ensure_resolved(df_path)
        self.df_path = df_path

        if exist_load and self.df_path.is_file():
            self.load()

    def __str__(self) -> str:
        return ''.join([
            self.__class__.__name__,
            '(',
            ', '.join([
                f'name={self.name}',
                str(dict(**self))                
            ]),
            ')'
        ])

    @property
    def name(self) -> str:
        return self._name

    def __getitem__(self, name: str) -> ExperimentVideo:
        return self._experiment_videos.__getitem__(name)

    def __setitem__(
            self,
            name: str,
            experiment_video: ExperimentVideo
    ) -> None:
        assert name == experiment_video.name, (
            'setting item must respect the experiment video name.')
        if not self._allow_key_overwrite and name in self:
            raise ValueError(
                f'overwriting existing element with name={name} with overwriting disabled.')
        self._experiment_videos.__setitem__(name, experiment_video)

    def __delitem__(self, name: str) -> None:
        self._experiment_videos.__delitem__(name)

    def __iter__(self) -> Iterator[ExperimentVideo]:
        return self._experiment_videos.__iter__()

    def __len__(self) -> int:
        return self._experiment_videos.__len__()

    def add(self, *experiment_videos: ExperimentVideo) -> None:
        for experiment_video in experiment_videos:
            self.__setitem__(experiment_video.name, experiment_video)

    def union(self, *others: Iterable[ExperimentVideo]) -> None:
        for other in others:
            self.add(*other)

    def discard(self, experiment_video: ExperimentVideo) -> None:
        self.__delitem__(experiment_video.name)

    @contextmanager
    def disable_key_overwriting(self) -> Iterator["ImageDataset"]:
        prev_state = self._allow_key_overwrite
        self._allow_key_overwrite = False
        yield self
        self._allow_key_overwrite = prev_state

    def video_paths(self) -> Iterable[Path]:
        return map(
            operator.attrgetter('video_path'),
            self.values()
        )

    def audio_paths(self) -> Iterable[Path]:
        return map(
            operator.attrgetter('audio_path'),
            self.values()
        )

    def frames_paths(self) -> Iterable[Path]:
        return map(
            operator.attrgetter('frames_path'),
            self.values()
        )

    def open_videos(self) -> None:
        for ev in self.values():
            ev.open_video()

    def extract_audios(
            self,
            overwrite: bool = False,
            verbose: VerboseType = False
    ) -> None:
        for experiment_video in self.values():
            experiment_video.extract_audio(
                overwrite=overwrite,
                verbose=verbose
            )

    def extract_frames(
            self,
            overwrite: bool = False,
            verbose: VerboseType = False,
            chunk_sizes: Optional[List[int]] = None,
            iterate: bool = True
    ) -> None:
        for experiment_video in self.values():
            experiment_video.extract_frames(
                chunk_sizes=chunk_sizes,
                prepend_name=True,
                iterate=iterate,
                overwrite=overwrite,
                verbose=verbose
            )

    def set_video_data(
            self,
            video_data: Mapping[str, Union[Mapping[str, Any], VideoData]],
            keys: ExperimentVideo.VideoDataKeys = ExperimentVideo.VideoDataKeys(),
            remove_absent: bool = False
    ) -> None:
        video_data_keys = frozenset(video_data.keys())
        self_keys = frozenset(self.keys())
        for name in self_keys & video_data_keys:
            self[name].set_video_data(
                video_data[name],
                keys
            )

        if remove_absent:
            for name in self_keys - video_data_keys:
                del self[name]

    def set_video_data_from_file(
            self,
            data_path: PathType,
            purge: bool = False,
            remove_absent: bool = False,
            keys: VideoDataKeys = VideoDataKeys()
    ) -> None:
        data_path = bl_utils.ensure_resolved(data_path)
        video_data = bl_io.load_json(data_path)
        purged = not purge

        if isinstance(video_data, list):
            if not purged:
                video_data = [
                    item
                    for item in video_data
                    if not item.pop(keys.ignore, False)
                ]
                purged = True

            video_data = {
                item.pop(keys.name): item
                for item in video_data
            }

        if isinstance(video_data, dict):
            if not purged:
                video_data = {
                    key: value
                    for key, value in video_data.items()
                    if not value.pop(keys.ignore, False)
                }

            self.set_video_data(video_data, keys, remove_absent=remove_absent)
        else:
            raise RuntimeError(f'could not load video data from {data_path}.')

    def load(
            self,
            path: Optional[PathType] = None,
            columns: Optional[Iterable[str]] = None
    ) -> None:
        if path is None:
            path = self.df_path
        else:
            self.df_path = bl_utils.ensure_resolved(path)

        if columns is None:
            self.df = pd.read_csv(self.df_path, skipinitialspace=True)
        else:
            self.df = pd.read_csv(
                self.df_path,
                skipinitialspace=True,
                usecols=tuple(columns)
            )

    def save(
            self,
            path: Optional[PathType] = None,
            overwrite: bool = False
    ) -> None:
        if path is None:
            path = self.df_path
        path = bl_utils.ensure_parent(path)

        if overwrite or not path.is_file():
            self.df.to_csv(path, index=False)

    def save_dfs(
            self,
            overwrite: bool = False
    ) -> None:
        bl_utils.functional.apply(
            operator.methodcaller('save_df', overwrite=overwrite),
            self.values()
        )

    def load_dfs(
            self,
            columns: Optional[Iterable[str]] = None,
            overwrite: bool = False,
            missing_ok: bool = False
    ) -> None:
        if columns is not None:
            columns = tuple(columns)

        bl_utils.functional.apply(
            operator.methodcaller(
                'load_df',
                columns=columns,
                overwrite=overwrite,
                missing_ok=missing_ok,
                inplace=True
            ),
            self.values()
        )

    def move(
            self,
            path: Union[str, bl_utils.PathType],
            renaming: bool = False,
            erase_old: bool = False,
            overwrite: bool = False
    ) -> None:
        if erase_old:
            old_path = self.df_path

        if renaming:
            self.df_path = self.df_path.with_name(path)
        else:
            self.df_path = bl_utils.ensure_resolved(path)

        self.save(overwrite=overwrite)

        if erase_old and old_path.is_file():
            old_path.unlink()
        # if erase: # Python 3.8 only
        #     old_path.unlink(missing_ok=True)

    @property
    def paths(self) -> pd.Series:
        return self.df[self.column_names.path]

    @paths.setter
    def paths(self, other: pd.Series) -> None:
        self.df[self.column_names.path] = other

    @overload
    def modify_path(
        self,
        old_path: PathType,
        new_path: PathType,
        many: bool # many: Literal[False]
    ) -> None: ...

    @overload
    def modify_path(
        self,
        old_path: Iterable[PathType],
        new_path: Iterable[PathType],
        many: bool # many: Literal[True]
    ) -> None: ...

    def modify_path(self, old_path, new_path, many):
        if many:
            old_to_new = dict(zip(map(Path, old_path), map(Path, new_path)))
            self.paths = self.df[self.column_names.path].apply(
                lambda x: old_to_new.get(Path(x), x)
            )
        else:
            self.paths = self.df[self.column_names.path].mask(
                lambda x: Path(x) == Path(old_path),
                new_path
            )

    def make_dataframe(
            self,
            recalculate: bool = False,
            exist_load: bool = False,
            enforce_time: bool = False,
            categories_as_int: bool = False,
            inplace: bool = True
    ) -> pd.DataFrame:
        dfs = map(
            operator.methodcaller(
                'make_dataframe',
                recalculate=recalculate,
                exist_load=exist_load,
                enforce_time=enforce_time,
                categories_as_int=categories_as_int,
                inplace=inplace
            ),
            self.values()
        )
        df = bl_utils.concatenate_dataframes(dfs)
        return df

    @overload
    def iterdata_from_dataframe(self, select_columns: str) -> Iterable[Tuple[np.ndarray, Any]]: ...

    @overload
    def iterdata_from_dataframe(self, select_columns: Optional[List[str]]) -> Iterable[Tuple[np.ndarray, dict]]: ...

    def iterdata_from_dataframe(self, select_columns=None):
        return itertools.chain.from_iterable(
            map(
                operator.methodcaller('iterdata_from_dataframe', select_columns),
                self.values()
            )
        )

    def as_tf_dataset(
            self,
            select_columns: Optional[Union[str, List[str]]] = None,
            inplace: bool = False
    ) -> tf.data.Dataset:
        datasets = collections.deque(map(
            operator.methodcaller('as_tf_dataset', select_columns, inplace=inplace),
            self.values()
        ))

        if not datasets:
            raise ValueError('resulting tensorflow dataset is empty.')

        ds = datasets.popleft()
        for dataset in datasets:
            ds = ds.concatenate(dataset)

        if inplace:
            self.ds = ds

        return ds

    def as_tf_dataset_dict(
            self,
            select_columns: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, tf.data.Dataset]:
        return {
            name: experiment_video.as_tf_dataset(select_columns=select_columns)
            for name, experiment_video in self.items()
        }
