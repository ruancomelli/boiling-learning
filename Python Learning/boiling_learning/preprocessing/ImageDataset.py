from dataclasses import dataclass
import itertools
import operator
from pathlib import Path
import typing
from typing import (
    overload,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union
)

import modin.pandas as pd
import numpy as np
import tensorflow as tf

from boiling_learning.utils import (PathType, VerboseType)
import boiling_learning.utils as bl_utils
import boiling_learning.io as bl_io
from boiling_learning.preprocessing.ExperimentVideo import ExperimentVideo

AUTOTUNE = tf.data.experimental.AUTOTUNE


class ImageDataset(
        bl_utils.SimpleRepr,
        bl_utils.SimpleStr,
        typing.MutableMapping[str, ExperimentVideo]
):
    '''
    TODO: improve this
    An ImageDataset is a file CSV in df_path and the correspondent images. The file in df_path contains at least two columns. One of this columns contains file paths, and the other the targets for training, validation or test. This is intended for using flow_from_dataframe. There may be an optional column which specifies if that image belongs to the training, the validation or the test sets.
    '''

    VideoData = ExperimentVideo.VideoData
    DataFrameColumnNames = ExperimentVideo.DataFrameColumnNames
    DataFrameColumnTypes = ExperimentVideo.DataFrameColumnTypes

    @dataclass(frozen=True)
    class VideoDataKeys(ExperimentVideo.VideoDataKeys):
        name: str = 'name'
        ignore: str = 'ignore'

    def __init__(
        self,
        column_names: DataFrameColumnNames = DataFrameColumnNames(),
        column_types: DataFrameColumnTypes = DataFrameColumnTypes(),
        df_path: Optional[PathType] = None,
        df: Optional[pd.DataFrame] = None,
        exist_load: bool = False
    ):
        if exist_load and df is not None:
            raise ValueError(
                'incompatible parameters: *df* and *exist_load=True*. Omit one of them.')

        self.column_names: self.DataFrameColumnNames = column_names
        self.column_types = column_types
        self._experiment_videos: Dict[str, ExperimentVideo] = {}

        if df_path is not None:
            df_path = bl_utils.ensure_resolved(df_path)
        self.df_path = df_path

        if exist_load and self.df_path.is_file():
            self.load()
        else:
            if df is None:
                df = pd.DataFrame()
            self.df = df

    def __getitem__(self, name: str) -> ExperimentVideo:
        return self._experiment_videos.__getitem__(name)

    def __setitem__(
            self,
            name: str,
            experiment_video: ExperimentVideo
    ) -> None:
        assert name == experiment_video.name, (
            'setting item must respect the experiment video name.')
        self._experiment_videos.__setitem__(name, experiment_video)

    def __delitem__(self, name: str) -> None:
        self._experiment_videos.__delitem__(name)

    def __iter__(self) -> Iterator[ExperimentVideo]:
        return self._experiment_videos.__iter__()

    def __len__(self) -> int:
        return self._experiment_videos.__len__()

    def add(self, experiment_video: ExperimentVideo) -> None:
        self.__setitem__(experiment_video.name, experiment_video)

    def discard(self, experiment_video: ExperimentVideo) -> None:
        self.__delitem__(experiment_video.name)

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
            keys: ExperimentVideo.VideoDataKeys
    ) -> None:
        common_names = frozenset(video_data) & frozenset(self)
        for name in common_names:
            self[name].set_video_data(
                video_data[name],
                keys
            )

    def set_video_data_from_file(
            self,
            data_path: PathType,
            purge: bool = False,
            keys: VideoDataKeys = VideoDataKeys(),
    ) -> None:
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

            self.set_video_data(video_data, keys)

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
                lambda y: old_to_new.get(Path(y), y)
            )
        else:
            self.paths = self.df[self.column_names.path].mask(
                lambda x: Path(x) == Path(old_path),
                new_path
            )

    def as_dataframe(self) -> pd.DataFrame:
        dfs = map(
            operator.methodcaller(
                'as_dataframe',
                self.column_names,
                self.column_types
            ),
            self.values()
        )

        self.df = bl_utils.concatenate_dataframes(dfs)

        return self.df

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
            column_spec: Union[Tuple[str, tf.DType], List[Tuple[str, tf.DType]]]
    ) -> tf.data.Dataset:
        if isinstance(column_spec[0], str):
            select_columns = column_spec[0]
            type_spec = column_spec[1]
        else:
            select_columns = [
                col_spec[0]
                for col_spec in column_spec
            ]
            type_spec = column_spec

        return tf.data.Dataset.from_generator(
            self.iterdata_from_dataframe,
            (tf.float32, type_spec),
            args=[select_columns]
        )


bl_utils.simple_pprint_class(ImageDataset)
