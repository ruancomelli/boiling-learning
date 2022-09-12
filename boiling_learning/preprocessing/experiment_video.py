import contextlib
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Union

import funcy
import modin.pandas as pd
from loguru import logger

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.describe.describers import describe
from boiling_learning.io import json
from boiling_learning.preprocessing.video import Video, VideoFrame, convert_video
from boiling_learning.utils.dataclasses import dataclass, field
from boiling_learning.utils.pathutils import PathLike, resolve


class ExperimentVideo:
    @dataclass
    class VideoData:
        '''Class for video data representation.
        # TODO: improve this doc

        Attributes
        ----------
        categories: [...]. Example: {
                'wire': 'NI80-...',
                'nominal_power': 85
            }
        fps: [...]. Example: 30
        ref_index: [...]. Example: 155
        ref_elapsed_time: [...]. Example: 12103
        '''

        categories: Mapping[str, Any] = field(default_factory=dict)
        fps: Optional[float] = None
        ref_index: Optional[int] = None
        ref_elapsed_time: Optional[timedelta] = None
        start_elapsed_time: Optional[timedelta] = None
        start_index: Optional[int] = None
        end_elapsed_time: Optional[timedelta] = None
        end_index: Optional[int] = None

    @dataclass(frozen=True)
    class VideoDataKeys:
        categories: str = 'categories'
        fps: str = 'fps'
        ref_index: str = 'ref_index'
        ref_elapsed_time: str = 'ref_elapsed_time'
        start_elapsed_time: str = 'start_elapsed_time'
        start_index: str = 'start_index'
        end_elapsed_time: str = 'end_elapsed_time'
        end_index: str = 'end_index'

    @dataclass(frozen=True)
    class DataFrameColumnNames:
        index: str = 'index'
        name: str = 'name'
        elapsed_time: str = 'elapsed_time'

    @dataclass(frozen=True)
    class DataFrameColumnTypes:
        index = int
        path = str
        name = str
        elapsed_time = 'timedelta64[s]'
        categories = 'category'

    def __init__(
        self,
        video_path: PathLike,
        name: str = '',
        df_dir: Optional[PathLike] = None,
        df_suffix: str = '.csv',
        df_path: Optional[PathLike] = None,
        column_names: DataFrameColumnNames = DataFrameColumnNames(),
        column_types: DataFrameColumnTypes = DataFrameColumnTypes(),
    ) -> None:
        self.path = resolve(video_path)
        self.video: SliceableDataset[VideoFrame] = Video(self.path)

        self._data: Optional[ExperimentVideo.VideoData] = None
        self.column_names = column_names
        self.column_types = column_types
        self.df: Optional[pd.DataFrame] = None
        self._name = name or self.path.stem

        if None not in {df_dir, df_path}:
            raise ValueError('at most one of (df_dir, df_path) must be given.')

        if not df_suffix.startswith('.'):
            raise ValueError('argument *df_suffix* must start with a dot \'.\'')

        self.df_path = (
            resolve(df_path)
            if df_path is not None
            else (
                (resolve(df_dir) / self.name).with_suffix(df_suffix)
                if df_dir is not None
                else None
            )
        )

    def __len__(self) -> int:
        return len(self.video)

    def __iter__(self) -> Iterator[VideoFrame]:
        return iter(self.video)

    def __str__(self) -> str:
        kwargs = {
            'name': self.name,
            'video_path': self.path,
            'df_path': self.df_path,
            'data': self.data,
            'column_names': self.column_names,
            'column_types': self.column_types,
        }
        joined_kwargs = ', '.join(f'{k}={v}' for k, v in kwargs.items() if v is not None)
        return f'{self.__class__.__name__}({joined_kwargs})'

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, path: PathLike) -> None:
        self._path = resolve(path)
        self.video = Video(self._path)

    @property
    def data(self) -> Optional[VideoData]:
        return self._data

    @data.setter
    def data(self, data: VideoData) -> None:
        self._data = data
        self._shrink_video_to_data()

    def _shrink_video_to_data(self) -> None:
        video_data = self.data

        assert video_data is not None

        logger.debug(f"Shrinking video to data for EV \"{self.name}\"")

        if video_data.start_index is not None:
            start = video_data.start_index
        elif video_data.start_elapsed_time is not None:
            assert video_data.fps is not None

            start = round(video_data.start_elapsed_time.total_seconds() * video_data.fps)
        else:
            start = 0

        if video_data.end_index is not None:
            end = video_data.end_index
        elif video_data.end_elapsed_time is not None:
            assert video_data.fps is not None

            end = round(video_data.end_elapsed_time.total_seconds() * video_data.fps)
        else:
            end = None

        if start != 0 or end is not None:
            self.video = self.video[start:end]

        logger.debug(f"Video in range {start}-{end} for EV \"{self.name}\"")

    def convert_video(
        self,
        dest_path: PathLike,
        overwrite: bool = False,
    ) -> None:
        """Use this function to move or convert video"""
        dest_path = resolve(dest_path, parents=True)
        convert_video(self.path, dest_path, overwrite=overwrite)
        self.path = dest_path

    def convert_dataframe_type(self, df: pd.DataFrame) -> pd.DataFrame:
        video_data = self.data

        assert video_data is not None

        col_types = funcy.merge(
            dict.fromkeys(video_data.categories, 'category'),
            {
                self.column_names.index: self.column_types.index,
                self.column_names.name: self.column_types.name,
                # self.column_names.elapsed_time: self.column_types.elapsed_time
                # BUG: including the line above rounds elapsed time, breaking the whole pipeline
            },
        )
        col_types = funcy.select_keys(set(df.columns), col_types)
        df = df.astype(col_types)

        with contextlib.suppress(KeyError):
            elapsed_time_column = self.column_names.elapsed_time
            if df[elapsed_time_column].dtype.kind == 'm':
                df[elapsed_time_column] = df[elapsed_time_column].dt.total_seconds()

        return df

    def make_dataframe(
        self,
        exist_load: bool = False,
        enforce_time: bool = False,
        inplace: bool = True,
    ) -> pd.DataFrame:
        if self.df_path is None:
            raise ValueError('*df_path* is not defined yet.')

        if self.df is not None:
            return self.df

        if exist_load and self.df_path.is_file():
            self.load_df()
            return self.df

        if self.data is None:
            raise ValueError('cannot convert to DataFrame. Video data must be previously set.')

        indices = range(len(self))

        data = {
            **self.data.categories,
            self.column_names.name: self.name,
            self.column_names.index: list(indices),
        }

        if (
            self.data.fps is not None
            and self.data.ref_index is not None
            and self.data.ref_elapsed_time is not None
        ):
            ref_index = self.data.ref_index
            ref_elapsed_time = pd.to_timedelta(self.data.ref_elapsed_time, unit='s')
            delta = pd.to_timedelta(1 / self.data.fps, unit='s')
            elapsed_time_list = [
                ref_elapsed_time + delta * (index - ref_index) for index in indices
            ]

            data[self.column_names.elapsed_time] = elapsed_time_list
        elif enforce_time:
            raise ValueError(
                'there is not enough time info in video data'
                ' (set *enforce_time*=False to suppress this error).'
            )

        df = pd.DataFrame(data)
        df = self.convert_dataframe_type(df)

        if inplace:
            self.df = df

        return df

    def sync_time_series(self, source_df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        df = self.make_dataframe(enforce_time=True, inplace=inplace)

        df = _sync_dataframes(
            source_df=source_df,
            dest_df=df,
            dest_time_column=self.column_names.elapsed_time,
        )

        if inplace:
            self.df = df

        return df

    def load_df(
        self,
        path: Optional[PathLike] = None,
        columns: Optional[Iterable[str]] = None,
        overwrite: bool = False,
        missing_ok: bool = False,
        inplace: bool = True,
    ) -> Optional[pd.DataFrame]:
        if path is not None:
            self.df_path = resolve(path)
        elif self.df_path is None:
            raise ValueError('*df_path* is not defined yet, so *path* must be given as argument.')

        logger.debug(
            f'Loading dataframe for experiment video {self.name} from file {self.df_path}'
        )

        if not overwrite and self.df is not None:
            return self.df

        if missing_ok and not self.df_path.is_file():
            return None

        df = pd.read_csv(
            self.df_path,
            skipinitialspace=True,
            usecols=tuple(columns) if columns is not None else None,
        )

        if inplace:
            self.df = df

        return df

    def save_df(self, path: Optional[PathLike] = None, overwrite: bool = False) -> None:
        if self.df is None:
            raise ValueError('*df* is not defined.')

        if path is None:
            if self.df_path is None:
                raise ValueError(
                    '*df_path* is not defined yet, so *path* must be given as argument.'
                )
            path = self.df_path

        path = resolve(path, parents=True)

        if overwrite or not path.is_file():
            self.df.to_csv(path, index=False)

    def targets(self, select_columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        df = self.make_dataframe()
        df = self.convert_dataframe_type(df)
        df.sort_values(by=self.column_names.index, inplace=True)

        if select_columns is not None:
            df = df[select_columns]

        return df


@json.encode.instance(ExperimentVideo)
def _encode_video(obj: ExperimentVideo) -> json.JSONDataType:
    return json.serialize(obj.path)


@describe.instance(ExperimentVideo)
def _describe_video(obj: ExperimentVideo) -> Path:
    return obj.path


def _sync_dataframes(
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

    if type(source_df.index) is not type(dest_df.index):  # noqa: E721 # do not compare types
        raise ValueError(
            f'the source and dest DataFrames indices must have the same type.'
            f' Got {type(source_df.index)} and {type(dest_df.index)}'
        )

    concat = pd.concat([source_df, dest_df]).sort_index()
    interpolation_method = 'index' if isinstance(source_df.index, pd.Float64Index) else 'time'
    concat = concat.interpolate(method=interpolation_method, limit_direction='both')
    concat = concat.loc[dest_df.index]
    return concat
