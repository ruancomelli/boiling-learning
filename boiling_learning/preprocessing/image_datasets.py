from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional, Type, Union

import funcy
import modin.pandas as pd
from dataclassy import dataclass

from boiling_learning.io import json
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.utils import PathLike, concatenate_dataframes, resolve, simple_pprint_class
from boiling_learning.utils.collections import KeyedSet
from boiling_learning.utils.descriptions import describe


@simple_pprint_class
class ImageDataset(KeyedSet[str, ExperimentVideo]):
    '''
    TODO: improve this
    An ImageDataset is a file CSV in df_path and the correspondent images. The file in df_path
    contains at least two columns. One of this columns contains file paths, and the other the
    targets for training, validation or test. This is intended for using flow_from_dataframe. There
    may be an optional column which specifies if that image belongs to the training, the validation
    or the test sets.
    '''

    VideoData: Type[ExperimentVideo.VideoData] = ExperimentVideo.VideoData
    DataFrameColumnNames: Type[
        ExperimentVideo.DataFrameColumnNames
    ] = ExperimentVideo.DataFrameColumnNames
    DataFrameColumnTypes: Type[
        ExperimentVideo.DataFrameColumnTypes
    ] = ExperimentVideo.DataFrameColumnTypes

    @dataclass(frozen=True, kwargs=True)
    class VideoDataKeys(ExperimentVideo.VideoDataKeys):
        name: str = 'name'
        ignore: str = 'ignore'

    def __init__(
        self,
        name: str,
        column_names: DataFrameColumnNames = DataFrameColumnNames(),
        column_types: DataFrameColumnTypes = DataFrameColumnTypes(),
        df_path: Optional[PathLike] = None,
        exist_load: bool = False,
    ) -> None:
        super().__init__(_get_experiment_video_name)

        self._name: str = name
        self.column_names: self.DataFrameColumnNames = column_names
        self.column_types: self.DataFrameColumnTypes = column_types
        self.df: Optional[pd.DataFrame] = None
        self.ds = None

        if df_path is not None:
            df_path = resolve(df_path)
        self.df_path = df_path

        if exist_load and self.df_path.is_file():
            self.load()

    def __str__(self) -> str:
        return ''.join(
            [
                self.__class__.__name__,
                '(',
                ', '.join(
                    [
                        f'name={self.name}',
                        f'column_names={self.column_names}',
                        f'column_types={self.column_types}',
                        f'df_path={self.df_path}',
                        f'experiment_videos={tuple(self.keys())}',
                    ]
                ),
                ')',
            ]
        )

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def make_union(
        cls,
        *others: ImageDataset,
        namer: Callable[[List[str]], str] = '+'.join,
    ) -> ImageDataset:
        name = namer(funcy.lpluck_attr('name', others))
        example = others[0]
        image_dataset = ImageDataset(name, example.column_names, example.column_types)
        image_dataset.update(*others)
        return image_dataset

    def open_videos(self) -> None:
        for ev in self:
            ev.open()

    def set_video_data(
        self,
        video_data: Mapping[str, Union[Mapping[str, Any], VideoData]],
        keys: ExperimentVideo.VideoDataKeys = ExperimentVideo.VideoDataKeys(),
        remove_absent: bool = False,
    ) -> None:
        video_data_keys = frozenset(video_data.keys())
        self_keys = frozenset(self.keys())
        for name in self_keys & video_data_keys:
            self[name].set_video_data(video_data[name], keys)

        if remove_absent:
            for name in self_keys - video_data_keys:
                del self[name]

    def set_video_data_from_file(
        self,
        data_path: PathLike,
        purge: bool = False,
        remove_absent: bool = False,
        keys: VideoDataKeys = VideoDataKeys(),
    ) -> None:
        data_path = resolve(data_path)
        video_data = _json.loads(data_path.read_text())
        purged = not purge

        if isinstance(video_data, list):
            if not purged:
                video_data = [item for item in video_data if not item.pop(keys.ignore, False)]
                purged = True

            video_data = {item.pop(keys.name): item for item in video_data}

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
        path: Optional[PathLike] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> None:
        if path is None:
            path = self.df_path
        else:
            self.df_path = resolve(path)

        self.df = pd.read_csv(
            self.df_path,
            skipinitialspace=True,
            usecols=tuple(columns) if columns is not None else None,
        )

    def save(self, path: Optional[PathLike] = None, overwrite: bool = False) -> None:
        if path is None:
            path = self.df_path
        path = resolve(path, parents=True)

        if overwrite or not path.is_file():
            self.df.to_csv(path, index=False)

    def save_dfs(self, overwrite: bool = False) -> None:
        for ev in self:
            ev.save_df(overwrite=overwrite)

    def load_dfs(
        self,
        columns: Optional[Iterable[str]] = None,
        overwrite: bool = False,
        missing_ok: bool = False,
    ) -> None:
        if columns is not None:
            columns = tuple(columns)

        for ev in self.values():
            ev.load_df(
                columns=columns,
                overwrite=overwrite,
                missing_ok=missing_ok,
                inplace=True,
            )

    def make_dataframe(
        self,
        recalculate: bool = False,
        exist_load: bool = False,
        enforce_time: bool = False,
        categories_as_int: bool = False,
        inplace: bool = True,
    ) -> pd.DataFrame:
        return concatenate_dataframes(
            ev.make_dataframe(
                recalculate=recalculate,
                exist_load=exist_load,
                enforce_time=enforce_time,
                categories_as_int=categories_as_int,
                inplace=inplace,
            )
            for ev in self
        )


def _get_experiment_video_name(experiment_video: ExperimentVideo) -> str:
    return experiment_video.name


@json.encode.instance(ImageDataset)
def _encode_image_dataset(obj: ImageDataset) -> List[json.JSONDataType]:
    return json.serialize(list(obj))


@describe.instance(ImageDataset)
def _describe_image_dataset(obj: ImageDataset) -> List[Path]:
    return describe(list(obj))
