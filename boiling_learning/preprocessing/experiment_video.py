from __future__ import annotations

import contextlib
import typing
from collections.abc import Iterable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

import funcy
import pandas as pd
from loguru import logger

from boiling_learning.dataclasses import field
from boiling_learning.datasets.sliceable import SliceableDataset
from boiling_learning.descriptions import describe
from boiling_learning.io import json
from boiling_learning.io.dataclasses import dataclass
from boiling_learning.preprocessing.video import Video, VideoFrame, convert_video
from boiling_learning.utils.pathutils import PathLike, resolve


@dataclass
class VideoData:
    """Data class for storing metadata and timing information about experiment videos.

    This class represents metadata associated with experiment videos, including
    categorical information about the experiment conditions and timing data for
    video synchronization and frame selection.

    Attributes
    ----------
    categories : Mapping[str, Any]
        Dictionary containing categorical information about the experiment,
        such as wire type and nominal power settings.
        Example: {
            'wire': 'NI80-...',
            'nominal_power': 85
        }
    fps : float | None
        Frames per second of the video. Used for time calculations and synchronization.
        Example: 30
    ref_index : int | None
        Reference frame index used for time synchronization.
        Example: 155
    ref_elapsed_time : timedelta | None
        Reference time point used for synchronization, in timedelta format.
        Example: timedelta(seconds=12103)
    start_elapsed_time : timedelta | None
        Start time of the video segment in timedelta format.
    start_index : int | None
        Starting frame index of the video segment.
    end_elapsed_time : timedelta | None
        End time of the video segment in timedelta format.
    end_index : int | None
        Ending frame index of the video segment.
    """

    categories: Mapping[str, Any] = field(default_factory=dict)
    fps: float | None = None
    ref_index: int | None = None
    ref_elapsed_time: timedelta | None = None
    start_elapsed_time: timedelta | None = None
    start_index: int | None = None
    end_elapsed_time: timedelta | None = None
    end_index: int | None = None

    def video_limits(self) -> tuple[int, int | None]:
        """Calculate the start and end frame indices for video processing.

        This method determines the frame range for video processing based on either:
        - Frame indices (start_index, end_index)
        - Elapsed time (start_elapsed_time, end_elapsed_time)

        The method prioritizes frame indices over elapsed time if both are provided.
        If neither is provided for a limit, it defaults to:
        - start: 0 (first frame)
        - end: None (process until the end of video)

        Returns:
            A tuple containing:
                - start: The starting frame index (inclusive)
                - end: The ending frame index (inclusive) or None to process until the end
        """
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
    index: str = "index"
    name: str = "name"
    elapsed_time: str = "elapsed_time"


@dataclass(frozen=True)
class _DataFrameColumnTypes:
    index = int
    path = str
    name = str
    elapsed_time = "timedelta64[s]"
    categories = "category"


_COLUMN_NAMES = _DataFrameColumnNames()
_COLUMN_TYPES = _DataFrameColumnTypes()


class ExperimentVideo:
    @dataclass(frozen=True)
    class VideoDataKeys:
        categories: str = "categories"
        fps: str = "fps"
        ref_index: str = "ref_index"
        ref_elapsed_time: str = "ref_elapsed_time"
        start_elapsed_time: str = "start_elapsed_time"
        start_index: str = "start_index"
        end_elapsed_time: str = "end_elapsed_time"
        end_index: str = "end_index"

    def __init__(
        self,
        video_path: PathLike,
        df_path: PathLike,
        name: str = "",
        data: VideoData | None = None,
        df: pd.DataFrame | None = None,
    ) -> None:
        self._path = resolve(video_path)
        self._video = Video(self.path)

        self._data = data
        self._name = name or self.path.stem

        self.df = df
        self.df_path = resolve(df_path)

    def with_data(self, data: VideoData) -> ExperimentVideo:
        return ExperimentVideo(
            self.path,
            self.df_path,
            name=self.name,
            data=data,
            df=self.df,
        )

    def __str__(self) -> str:
        kwargs = {
            "name": self.name,
            "video_path": self.path,
            "df_path": self.df_path,
            "data": self.data,
        }
        joined_kwargs = ", ".join(
            f"{k}={v}" for k, v in kwargs.items() if v is not None
        )
        return f"{self.__class__.__name__}({joined_kwargs})"

    @property
    def video(self) -> Video:
        return self._video

    @property
    def data(self) -> VideoData | None:
        return self._data

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        return self._path

    @property
    def start(self) -> int:
        if self.data is None:
            return 0

        start, _ = self.data.video_limits()
        return start

    @property
    def end(self) -> int:
        if self.data is not None:
            _, end = self.data.video_limits()

            if end is not None:
                return end

        return len(self._video)

    def frames(self) -> SliceableDataset[VideoFrame]:
        return self._video[self.start : self.end]

    def convert_video(
        self,
        dest_path: PathLike,
        overwrite: bool = False,
    ) -> None:
        """Use this function to move or convert video."""
        dest_path = resolve(dest_path, parents=True)
        convert_video(self.path, dest_path, overwrite=overwrite)

    def make_dataframe(self, *, enforce_time: bool = False) -> pd.DataFrame:
        if self.df is not None:
            return self.df

        if self.data is None:
            raise ValueError(
                "cannot convert to DataFrame. Video data must be previously set."
            )

        indices = range(len(self.frames()))

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
            ref_elapsed_time = pd.to_timedelta(self.data.ref_elapsed_time, unit="s")
            delta = pd.to_timedelta(1 / self.data.fps, unit="s")
            elapsed_time_list = [
                ref_elapsed_time + delta * (index - ref_index) for index in indices
            ]

            data[_COLUMN_NAMES.elapsed_time] = elapsed_time_list
        elif enforce_time:
            raise ValueError(
                "there is not enough time info in video data"
                " (set `enforce_time=False` to suppress this error)."
            )

        df = pd.DataFrame(data)
        df = _convert_dataframe_type(df, self.data.categories)

        return df

    def sync_time_series(self, source_df: pd.DataFrame) -> pd.DataFrame:
        df = self.make_dataframe(enforce_time=True)

        df = _sync_dataframes(
            source_df=source_df,
            dest_df=df,
            dest_time_column=_COLUMN_NAMES.elapsed_time,
        )

        return df

    def load_df(self) -> pd.DataFrame:
        logger.debug(
            f"Loading dataframe for experiment video {self.name} from file {self.df_path}"
        )

        if self.df is not None:
            return self.df

        return typing.cast(
            pd.DataFrame,
            pd.read_csv(self.df_path, skipinitialspace=True),
        )

    def save_df(self, df: pd.DataFrame, /) -> None:
        path = resolve(self.df_path, parents=True)
        df.to_csv(path, index=False)

    def targets(self) -> pd.DataFrame:
        assert self.data is not None

        df = self.make_dataframe()
        df = _convert_dataframe_type(df, self.data.categories)
        df.sort_values(by=_COLUMN_NAMES.index, inplace=True)

        return df


@json.encode.instance(ExperimentVideo)
def _encode_video(obj: ExperimentVideo) -> json.JSONDataType:
    return json.serialize(obj.path)


@describe.instance(ExperimentVideo)
def _describe_video(obj: ExperimentVideo) -> Path:
    return obj.path


def _convert_dataframe_type(
    df: pd.DataFrame, categories: Iterable[str]
) -> pd.DataFrame:
    col_types = funcy.merge(
        dict.fromkeys(categories, "category"),
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
        if df[elapsed_time_column].dtype.kind == "m":
            df[elapsed_time_column] = df[elapsed_time_column].dt.total_seconds()

    return df


def _sync_dataframes(
    source_df: pd.DataFrame,
    dest_df: pd.DataFrame,
    dest_time_column: str | None = None,
) -> pd.DataFrame:
    allowed_index = (pd.DatetimeIndex, pd.TimedeltaIndex, pd.Float64Index)

    if not isinstance(source_df.index, allowed_index):
        raise ValueError(
            f"the source DataFrame index must be one of {allowed_index}."
            " Ensure this or pass a valid column name as input."
            f" Got {type(source_df.index)}"
        )

    if dest_time_column is not None:
        dest_df = dest_df.set_index(dest_time_column, drop=False)

    if not isinstance(dest_df.index, allowed_index):
        raise ValueError(
            f"the dest DataFrame index must be one of {allowed_index}."
            " Ensure this or pass a valid column name as input."
            f" Got {type(dest_df.index)}"
        )

    if isinstance(source_df.index, pd.TimedeltaIndex):
        source_df.index = source_df.index.total_seconds()

    if isinstance(dest_df.index, pd.TimedeltaIndex):
        dest_df.index = dest_df.index.total_seconds()

    if type(source_df.index) is not type(dest_df.index):
        raise ValueError(
            f"the source and dest DataFrames indices must have the same type."
            f" Got {type(source_df.index)} and {type(dest_df.index)}"
        )

    concat = pd.concat([source_df, dest_df]).sort_index()
    interpolation_method = (
        "index" if isinstance(source_df.index, pd.Float64Index) else "time"
    )
    concat = concat.interpolate(method=interpolation_method, limit_direction="both")
    concat = concat.loc[dest_df.index]
    return concat
