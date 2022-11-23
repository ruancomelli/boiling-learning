import contextlib
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import funcy
import modin.pandas as pd
from loguru import logger

from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.descriptions import describe
from boiling_learning.io import json
from boiling_learning.io.storage import dataclass
from boiling_learning.preprocessing.video import Video, VideoFrame, convert_video
from boiling_learning.utils.dataclasses import field
from boiling_learning.utils.pathutils import PathLike, resolve


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

    def video_limits(self) -> tuple[int, Optional[int]]:
        if self.start_index is not None:
            start = self.start_index
        elif self.start_elapsed_time is not None:
            assert self.fps is not None

            start = round(self.start_elapsed_time.total_seconds() * self.fps)
        else:
            start = 0

        if self.end_index is not None:
            end = self.end_index
        elif self.end_elapsed_time is not None:
            assert self.fps is not None

            end = round(self.end_elapsed_time.total_seconds() * self.fps)
        else:
            end = None

        return start, end


@dataclass(frozen=True)
class _DataFrameColumnNames:
    index: str = 'index'
    name: str = 'name'
    elapsed_time: str = 'elapsed_time'


@dataclass(frozen=True)
class _DataFrameColumnTypes:
    index = int
    path = str
    name = str
    elapsed_time = 'timedelta64[s]'
    categories = 'category'


_COLUMN_NAMES = _DataFrameColumnNames()
_COLUMN_TYPES = _DataFrameColumnTypes()


class ExperimentVideo:
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

    def __init__(
        self,
        video_path: PathLike,
        df_path: PathLike,
        name: str = '',
    ) -> None:
        self._path = resolve(video_path)
        self.video: SliceableDataset[VideoFrame] = Video(self.path)

        self._data: Optional[VideoData] = None
        self._name = name or self.path.stem

        self.df: Optional[pd.DataFrame] = None
        self.df_path = resolve(df_path)

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
        }
        joined_kwargs = ', '.join(f'{k}={v}' for k, v in kwargs.items() if v is not None)
        return f'{self.__class__.__name__}({joined_kwargs})'

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        return self._path

    @property
    def data(self) -> Optional[VideoData]:
        return self._data

    @data.setter
    def data(self, data: VideoData) -> None:
        self._data = data

        logger.debug(f"Shrinking video to data for EV \"{self.name}\"")
        start, end = data.video_limits()

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

    def make_dataframe(
        self,
        *,
        enforce_time: bool = False,
        inplace: bool = True,
    ) -> pd.DataFrame:
        if self.df is not None:
            return self.df

        if self.data is None:
            raise ValueError('cannot convert to DataFrame. Video data must be previously set.')

        indices = range(len(self))

        data = {
            **self.data.categories,
            _COLUMN_NAMES.name: self.name,
            _COLUMN_NAMES.index: list(indices),
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

            data[_COLUMN_NAMES.elapsed_time] = elapsed_time_list
        elif enforce_time:
            raise ValueError(
                'there is not enough time info in video data'
                ' (set *enforce_time*=False to suppress this error).'
            )

        df = pd.DataFrame(data)
        df = _convert_dataframe_type(df, self.data.categories)

        if inplace:
            self.df = df

        return df

    def sync_time_series(self, source_df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        df = self.make_dataframe(enforce_time=True, inplace=inplace)

        df = _sync_dataframes(
            source_df=source_df,
            dest_df=df,
            dest_time_column=_COLUMN_NAMES.elapsed_time,
        )

        if inplace:
            self.df = df

        return df

    def load_df(self, overwrite: bool = False, inplace: bool = True) -> pd.DataFrame:
        logger.debug(
            f'Loading dataframe for experiment video {self.name} from file {self.df_path}'
        )

        if not overwrite and self.df is not None:
            return self.df

        df: pd.DataFrame = pd.read_csv(self.df_path, skipinitialspace=True)

        if inplace:
            self.df = df

        return df

    def save_df(self, overwrite: bool = False) -> None:
        if self.df is None:
            raise ValueError('`df` is not defined')

        path = resolve(self.df_path, parents=True)

        if overwrite or not path.is_file():
            self.df.to_csv(path, index=False)

    def targets(self) -> pd.DataFrame:
        df = self.make_dataframe()

        assert self.data is not None
        df = _convert_dataframe_type(df, self.data.categories)
        df.sort_values(by=_COLUMN_NAMES.index, inplace=True)

        return df


@json.encode.instance(ExperimentVideo)
def _encode_video(obj: ExperimentVideo) -> json.JSONDataType:
    return json.serialize(obj.path)


@describe.instance(ExperimentVideo)
def _describe_video(obj: ExperimentVideo) -> Path:
    return obj.path


def _convert_dataframe_type(df: pd.DataFrame, categories: Iterable[str]) -> pd.DataFrame:
    col_types = funcy.merge(
        dict.fromkeys(categories, 'category'),
        {
            _COLUMN_NAMES.index: _COLUMN_TYPES.index,
            _COLUMN_NAMES.name: _COLUMN_TYPES.name,
            # _COLUMN_NAMES.elapsed_time: _COLUMN_TYPES.elapsed_time
            # BUG: including the line above rounds elapsed time, breaking the whole pipeline
        },
    )

    col_types = funcy.select_keys(set(df.columns), col_types)
    df = df.astype(col_types)

    with contextlib.suppress(KeyError):
        elapsed_time_column = _COLUMN_NAMES.elapsed_time
        if df[elapsed_time_column].dtype.kind == 'm':
            df[elapsed_time_column] = df[elapsed_time_column].dt.total_seconds()

    return df


def _sync_dataframes(
    source_df: pd.DataFrame,
    dest_df: pd.DataFrame,
    dest_time_column: Optional[str] = None,
) -> pd.DataFrame:
    allowed_index = (pd.DatetimeIndex, pd.TimedeltaIndex, pd.Float64Index)

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
