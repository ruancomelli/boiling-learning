import dataclasses
from dataclasses import dataclass
import datetime
import operator
from pathlib import Path
from typing import (
    overload,
    Any,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union
)

from scipy.interpolate import interp1d
import numpy as np
import modin.pandas as pd
import pims
from toolz import dicttoolz

from boiling_learning.preprocessing.video import (
    convert_video,
    extract_audio,
    chunked_filename_pattern,
    extract_frames
)
import boiling_learning.utils as bl_utils
from boiling_learning.utils import (
    PathType,
    VerboseType
)


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
        ref_image: [...]. Example: 'GOPR_frame1263.png'
        ref_elapsed_time: [...]. Example: 12103
        '''
        categories: Optional[Mapping[str, Any]] = dataclasses.field(default_factory=dict)
        fps: Optional[float] = None
        ref_frame: Optional[str] = None
        ref_elapsed_time: Optional[str] = None

    @dataclass(frozen=True)
    class VideoDataKeys:
        categories: str = 'categories'
        fps: str = 'fps'
        ref_frame: str = 'ref_frame'
        ref_elapsed_time: str = 'ref_elapsed_time'

    @dataclass(frozen=True)
    class DataFrameColumnNames:
        index: str = 'index'
        path: Optional[str] = None
        name: str = 'name'
        elapsed_time: str = 'elapsed_time'

    @dataclass(frozen=True)
    class DataFrameColumnTypes:
        index = int
        path = str
        name = str
        elapsed_time = float
        categories = 'category'

    def __init__(
            self,
            video_path: PathType,
            name: Optional[str] = None,
            frames_dir: Optional[PathType] = None,
            frames_path: Optional[PathType] = None,
            frames_suffix: str = '.png',
            audio_dir: Optional[PathType] = None,
            audio_path: Optional[PathType] = None,
            audio_suffix: str = '.m4a',
            column_names: DataFrameColumnNames = DataFrameColumnNames(),
            column_types: DataFrameColumnTypes = DataFrameColumnTypes()
    ):
        self.video_path: Path = bl_utils.ensure_resolved(video_path)
        self.frames_path: Path
        self.audio_path: Path
        self.data: Optional[self.VideoData] = None
        self._name: str
        self.column_names = column_names
        self.column_types = column_types
        self.df = None

        if name is None:
            self._name = self.video_path.stem
        else:
            self._name = name

        if (frames_dir is None) ^ (frames_path is None):
            self.frames_path = (
                bl_utils.ensure_resolved(frames_path)
                if frames_dir is None
                else bl_utils.ensure_resolved(frames_dir) / self.name
            )
        else:
            raise ValueError(
                'exactly one of (frames_dir, frames_path) must be given.')

        if frames_suffix.startswith('.'):
            self.frames_suffix = frames_suffix
        else:
            raise ValueError(
                'argument *frames_suffix* must start with a dot \'.\'')

        if not audio_suffix.startswith('.'):
            raise ValueError(
                'argument *audio_suffix* must start with a dot \'.\'')

        if (audio_dir is None) ^ (audio_path is None):
            self.audio_path = (
                bl_utils.ensure_resolved(audio_path)
                if audio_dir is None
                else (bl_utils.ensure_resolved(audio_dir) / self.name).with_suffix(audio_suffix)
            )
        else:
            raise ValueError(
                'exactly one of (audio_dir, audio_path) must be given.')

    @property
    def name(self) -> str:
        return self._name

    def convert_video(
            self,
            dest_path: PathType,
            overwrite: bool = False,
            verbose: VerboseType = False
    ) -> None:
        """Use this function to move or convert video
        """
        convert_video(
            self.video_path,
            dest_path,
            overwrite=overwrite,
            verbose=verbose
        )
        self.video_path = dest_path

    def extract_audio(
            self,
            overwrite: bool = False,
            verbose: VerboseType = False
    ) -> None:
        extract_audio(
            self.video_path,
            self.audio_path,
            overwrite=overwrite,
            verbose=verbose
        )

    def extract_frames(
            self,
            chunk_sizes: Optional[List[int]] = None,
            prepend_name: bool = True,
            iterate: bool = True,
            overwrite: bool = False,
            verbose: VerboseType = False
    ) -> None:
        filename_pattern = 'frame{index}' + self.frames_suffix
        if prepend_name:
            filename_pattern = '_'.join((self.name, filename_pattern))

        if chunk_sizes is not None:
            filename_pattern = chunked_filename_pattern(
                chunk_sizes=chunk_sizes,
                chunk_name='{min_index}-{max_index}',
                filename=filename_pattern
            )

        extract_frames(
            self.video_path,
            outputdir=self.frames_path,
            filename_pattern=filename_pattern,
            frame_suffix=self.frames_suffix,
            verbose=verbose,
            fast_frames_count=None if overwrite else not iterate,
            overwrite=overwrite,
            iterate=iterate
        )

    def _frame_stem_format(self) -> str:
        return self.name + '_frame{index}'

    def frame_stem(self, index: int) -> str:
        return self._frame_stem_format().format(index=index)

    def _frame_name_format(self) -> str:
        return self._frame_stem_format() + self.frames_suffix

    def frame_name(self, index: int) -> str:
        return self._frame_name_format().format(index=index)

    def frames(self) -> Sequence[np.ndarray]:
        return pims.Video(self.video_path)

    def glob_frames(self) -> Iterable[Path]:
        return self.frames_path.rglob('*' + self.frames_suffix)

    def set_video_data(
            self,
            data: Union[Mapping[str, Any], VideoData],
            keys: VideoDataKeys = VideoDataKeys()
    ) -> None:
        if isinstance(data, self.VideoData):
            self.data = data
        else:
            self.data = bl_utils.dataclass_from_mapping(
                data,
                self.VideoData,
                key_map=keys
            )

    def set_data(
            self,
            data_source: pd.DataFrame,
            source_time_column: str
    ) -> pd.DataFrame:
        '''Define data (other than the ones specified as video data) from a source *data_source*

        Example usage:
        >>>> data_source = pd.read_csv('my_data.csv')
        >>>> time_column, hf_column, temperature_column = 'time', 'heat_flux', 'temperature'
        >>>> ev.set_data(
            data_source[[time_column, hf_column, temperature_column]],
            source_time_column=time_column
        )

        WARNING: if *data_source* contains
        '''

        if self.data is None:
            raise ValueError('cannot set data while video_data is not set.')

        self.make_dataframe(recalculate=False, enforce_time=True)

        columns_to_set = tuple(x for x in data_source.columns if x != source_time_column)
        intersect = frozenset(columns_to_set) & frozenset(self.df.columns)
        if intersect:
            raise ValueError(
                f'the columns {intersect} exist both in *data_source* and in this dataframe.'
                ' Make sure you rename *data_source* columns to avoid this error.')

        time = data_source[source_time_column]
        for column in columns_to_set:
            interpolator = interp1d(time, data_source[column])
            self.df[column] = interpolator(self.df[self.column_names.elapsed_time])

        return self.df

    def make_dataframe(
            self,
            recalculate: bool = False,
            enforce_time: bool = False
    ) -> pd.DataFrame:
        if not recalculate and self.df is not None:
            return self.df

        if self.data is None:
            raise ValueError(
                'cannot convert to DataFrame. Video data must be previously set.')

        with self.frames() as f:
            indices = range(len(f))

        data = bl_utils.merge_dicts(
            {
                self.column_names.name: self.name,
                self.column_names.index: list(indices)
            },
            self.data.categories,
            latter_precedence=False
        )

        available_time_info = map(
            bl_utils.is_not(None),
            (
                self.data.fps,
                self.data.ref_frame,
                self.data.ref_elapsed_time
            )
        )
        if all(available_time_info):
            ref_index = self.data.ref_index
            delta = datetime.timedelta(seconds=1/self.data.fps)
            elapsed_time_list = [
                self.data.ref_elapsed_time + delta*(index - ref_index)
                for index in indices
            ]

            data[self.column_names.elapsed_time] = elapsed_time_list
        elif enforce_time:
            raise ValueError(
                'there is not enough time info in video data'
                ' (set *enforce_time*=False to suppress this error).')

        if self.column_names.path is not None:
            paths = sorted(
                self.glob_frames(),
                key=operator.attrgetter('stem')
            )
            data[self.column_names.path] = paths

        df = pd.DataFrame(data)

        col_types = bl_utils.merge_dicts(
            dict.fromkeys(self.data.categories, 'category'),
            {
                self.column_names.path: self.column_types.path,
                self.column_names.name: self.column_types.name,
                self.column_names.elapsed_time: self.column_types.elapsed_time
            }
        )
        col_types = dicttoolz.keyfilter(
            set(df.columns).__contains__,
            col_types
        )

        self.df = df.astype(col_types)

        return self.df

    @overload
    def iterdata_from_dataframe(self, select_columns: str) -> Iterable[Tuple[np.ndarray, Any]]: ...

    @overload
    def iterdata_from_dataframe(self, select_columns: Optional[List[str]]) -> Iterable[Tuple[np.ndarray, dict]]: ...

    def iterdata_from_dataframe(self, select_columns=None):
        df = self.make_dataframe(recalculate=False)
        indices = df[self.column_names.index]

        data = df
        if select_columns is not None:
            data = data[select_columns]
            if not isinstance(select_columns, str):
                data = data.to_dict(orient='records')

        with self.frames() as f:
            return zip(
                map(f.__getitem__, indices),
                data
            )
