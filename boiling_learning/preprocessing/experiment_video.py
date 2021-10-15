import operator
from datetime import timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import funcy
import modin.pandas as pd
import numpy as np
import tensorflow as tf
from dataclassy import dataclass
from scipy.interpolate import interp1d

import boiling_learning.utils as bl_utils
from boiling_learning.io.io import chunked_filename_pattern, load_dataset
from boiling_learning.preprocessing.preprocessing import sync_dataframes
from boiling_learning.preprocessing.Video import Video
from boiling_learning.preprocessing.video import (
    convert_video,
    extract_audio,
    extract_frames,
)
from boiling_learning.utils import PathLike, VerboseType, ensure_resolved
from boiling_learning.utils.dtypes import auto_spec
from boiling_learning.utils.slicerators import Slicerator


class ExperimentVideo(Video):
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

        categories: Mapping[str, Any] = {}
        fps: Optional[float] = None
        ref_index: Optional[int] = None
        ref_elapsed_time: Optional[timedelta] = None
        start_elapsed_time: Optional[timedelta] = None
        start_index: Optional[int] = None
        end_elapsed_time: Optional[timedelta] = None
        end_index: Optional[int] = None

    @dataclass(frozen=True, kwargs=True)
    class VideoDataKeys:
        categories: str = 'categories'
        fps: str = 'fps'
        ref_index: str = 'ref_index'
        ref_elapsed_time: str = 'ref_elapsed_time'
        start_elapsed_time: str = 'start_elapsed_time'
        start_index: str = 'start_index'
        end_elapsed_time: str = 'end_elapsed_time'
        end_index: str = 'end_index'

    @dataclass(frozen=True, kwargs=True)
    class DataFrameColumnNames:
        index: str = 'index'
        path: Optional[str] = None
        name: str = 'name'
        elapsed_time: str = 'elapsed_time'

    @dataclass(frozen=True, kwargs=True)
    class DataFrameColumnTypes:
        index = int
        path = str
        name = str
        elapsed_time = 'timedelta64[s]'
        categories = 'category'

    def __init__(
        self,
        video_path: PathLike,
        name: Optional[str] = None,
        frames_dir: Optional[PathLike] = None,
        frames_suffix: str = '.png',
        frames_path: Optional[PathLike] = None,
        frames_tensor_dir: Optional[PathLike] = None,
        frames_tensor_path: Optional[PathLike] = None,
        audio_dir: Optional[PathLike] = None,
        audio_suffix: str = '.m4a',
        audio_path: Optional[PathLike] = None,
        df_dir: Optional[PathLike] = None,
        df_suffix: str = '.csv',
        df_path: Optional[PathLike] = None,
        column_names: DataFrameColumnNames = DataFrameColumnNames(),
        column_types: DataFrameColumnTypes = DataFrameColumnTypes(),
    ) -> None:
        super().__init__(video_path)

        self.frames_path: Optional[Path]
        self.frames_suffix: str
        self.audio_path: Optional[Path]
        self.df_path: Optional[Path]
        self.frames_tensor_path: Optional[Path]
        self._data: Optional[self.VideoData] = None
        self._name: str
        self.column_names: self.DataFrameColumnNames = column_names
        self.column_types: self.DataFrameColumnTypes = column_types
        self.df: Optional[pd.DataFrame] = None
        self.ds: Optional[tf.data.Dataset] = None
        self._name: str = name if name is not None else self.path.stem

        if None not in {frames_dir, frames_path}:
            raise ValueError(
                'at most one of (*frames_dir*, *frames_path*) must be given.'
            )

        if None not in {frames_tensor_path, frames_tensor_dir}:
            raise ValueError(
                'at most one of (*frames_tensor_path*, *frames_tensor_dir*) '
                'must be given.'
            )

        if None not in {audio_dir, audio_path}:
            raise ValueError(
                'at most one of (*audio_dir*, *audio_path*) must be given.'
            )

        if None not in {df_dir, df_path}:
            raise ValueError('at most one of (df_dir, df_path) must be given.')

        if not frames_suffix.startswith('.'):
            raise ValueError(
                'argument *frames_suffix* must start with a dot \'.\''
            )

        if not audio_suffix.startswith('.'):
            raise ValueError(
                'argument *audio_suffix* must start with a dot \'.\''
            )

        if not df_suffix.startswith('.'):
            raise ValueError(
                'argument *df_suffix* must start with a dot \'.\''
            )

        self.frames_path: Optional[Path] = (
            ensure_resolved(frames_path)
            if frames_path is not None
            else (
                ensure_resolved(frames_dir) / self.name
                if frames_dir is not None
                else None
            )
        )

        self.frames_tensor_path: Optional[Path] = (
            ensure_resolved(frames_tensor_path)
            if frames_tensor_path is not None
            else (
                ensure_resolved(frames_tensor_dir) / self.name
                if frames_tensor_dir is not None
                else None
            )
        )

        self.frames_suffix: str = frames_suffix

        self.audio_path: Optional[Path] = (
            ensure_resolved(audio_path)
            if audio_path is not None
            else (
                (ensure_resolved(audio_dir) / self.name).with_suffix(
                    audio_suffix
                )
                if audio_dir is not None
                else None
            )
        )

        self.df_path: Optional[Path] = (
            ensure_resolved(df_path)
            if df_path is not None
            else (
                (ensure_resolved(df_dir) / self.name).with_suffix(df_suffix)
                if df_dir is not None
                else None
            )
        )

    def __str__(self) -> str:
        kwargs = {
            'name': self.name,
            'video_path': self.path,
            'frames_path': self.frames_path,
            'frames_suffix': self.frames_suffix,
            'audio_path': self.audio_path,
            'df_path': self.df_path,
            'frames_tensor_path': self.frames_tensor_path,
            'data': self.data,
            'column_names': self.column_names,
            'column_types': self.column_types,
        }

        return ''.join(
            (
                self.__class__.__name__,
                '(',
                ', '.join(
                    f'{k}={v}' for k, v in kwargs.items() if v is not None
                ),
                ')',
            )
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> Optional[VideoData]:
        return self._data

    @data.setter
    def data(self, data: VideoData) -> None:
        self._data = data

        if data.start_index is not None:
            self.start = data.start_index
        elif data.start_elapsed_time is not None:
            self.start = round(
                data.start_elapsed_time.total_seconds() * data.fps
            )

        if data.end_index is not None:
            self.end = data.end_index
        elif data.end_elapsed_time is not None:
            self.end = round(data.end_elapsed_time.total_seconds() * data.fps)

    def convert_video(
        self,
        dest_path: PathLike,
        overwrite: bool = False,
        verbose: VerboseType = False,
    ) -> None:
        """Use this function to move or convert video"""
        dest_path = bl_utils.ensure_parent(dest_path)
        convert_video(
            self.path, dest_path, overwrite=overwrite, verbose=verbose
        )
        self.path = dest_path

    def extract_audio(
        self, overwrite: bool = False, verbose: VerboseType = False
    ) -> None:
        if self.audio_path is None:
            raise ValueError('*audio_path* is not defined yet.')

        extract_audio(
            self.path,
            self.audio_path,
            overwrite=overwrite,
            verbose=verbose,
        )

    def extract_frames(
        self,
        chunk_sizes: Optional[List[int]] = None,
        prepend_name: bool = True,
        iterate: bool = True,
        overwrite: bool = False,
        verbose: VerboseType = False,
    ) -> None:
        if self.frames_path is None:
            raise ValueError('*frames_path* is not defined yet.')

        filename_pattern = 'frame{index}' + self.frames_suffix
        if prepend_name:
            filename_pattern = '_'.join((self.name, filename_pattern))

        if chunk_sizes is not None:
            filename_pattern = chunked_filename_pattern(
                chunk_sizes=chunk_sizes,
                chunk_name='{min_index}-{max_index}',
                filename=filename_pattern,
            )

        extract_frames(
            self.path,
            outputdir=self.frames_path,
            filename_pattern=filename_pattern,
            frame_suffix=self.frames_suffix,
            verbose=verbose,
            fast_frames_count=None if overwrite else not iterate,
            overwrite=overwrite,
            iterate=iterate,
        )

    def _frame_stem_format(self) -> str:
        return self.name + '_frame{index}'

    def frame_stem(self, index: int) -> str:
        return self._frame_stem_format().format(index=index)

    def _frame_name_format(self) -> str:
        return self._frame_stem_format() + self.frames_suffix

    def frame_name(self, index: int) -> str:
        return self._frame_name_format().format(index=index)

    def glob_frames(self) -> Iterable[Path]:
        if self.frames_path is None:
            raise ValueError('*frames_path* is not defined yet.')
        return self.frames_path.rglob('*' + self.frames_suffix)

    def set_video_data(
        self,
        data: Union[Mapping[str, Any], VideoData],
        keys: VideoDataKeys = VideoDataKeys(),
    ) -> None:
        if isinstance(data, self.VideoData):
            self.data = data
        else:
            self.data = bl_utils.dataclass_from_mapping(
                data, self.VideoData, key_map=keys
            )

    def set_data(
        self, data_source: pd.DataFrame, source_time_column: str
    ) -> pd.DataFrame:
        '''Define data (other than the ones specified as video data) from a source *data_source*

        Example usage:
        ```
        data_source = pd.read_csv('my_data.csv')
        time_column, hf_column, temperature_column = 'time', 'heat_flux', 'temperature'
        ev.set_data(
            data_source[[time_column, hf_column, temperature_column]],
            source_time_column=time_column
        )
        ```

        WARNING: if *data_source* contains
        '''
        self.make_dataframe(recalculate=False, enforce_time=True)

        columns_to_set = tuple(
            x for x in data_source.columns if x != source_time_column
        )
        intersect = frozenset(columns_to_set) & frozenset(self.df.columns)
        if intersect:
            raise ValueError(
                f'the columns {intersect} exist both in *data_source* and in this dataframe.'
                ' Make sure you rename *data_source* columns to avoid this error.'
            )

        time = data_source[source_time_column]
        for column in columns_to_set:
            interpolator = interp1d(time, data_source[column])
            self.df[column] = interpolator(
                self.df[self.column_names.elapsed_time]
            )

        return self.df

    def convert_dataframe_type(
        self, df: pd.DataFrame, categories_as_int: bool = False
    ) -> pd.DataFrame:
        col_types = funcy.merge(
            dict.fromkeys(self.data.categories, 'category'),
            {
                self.column_names.index: self.column_types.index,
                self.column_names.path: self.column_types.path,
                self.column_names.name: self.column_types.name,
                # self.column_names.elapsed_time: self.column_types.elapsed_time
                # BUG: including the line above rounds elapsed time, breaking the whole pipeline
            },
        )
        col_types = funcy.select_keys(set(df.columns), col_types)
        df = df.astype(col_types)

        try:
            if df[self.column_names.elapsed_time].dtype.kind == 'm':
                df[self.column_names.elapsed_time] = df[
                    self.column_names.elapsed_time
                ].dt.total_seconds()
        except KeyError:
            pass

        if categories_as_int:
            df = bl_utils.dataframe_categories_to_int(df, inplace=True)

        return df

    def make_dataframe(
        self,
        recalculate: bool = False,
        exist_load: bool = False,
        enforce_time: bool = False,
        categories_as_int: bool = False,
        inplace: bool = True,
    ) -> pd.DataFrame:
        if self.df_path is None:
            raise ValueError('*df_path* is not defined yet.')

        if not recalculate and self.df is not None:
            return self.df

        if exist_load and self.df_path.is_file():
            self.load_df()
            return self.df

        if self.data is None:
            raise ValueError(
                'cannot convert to DataFrame. Video data must be previously set.'
            )

        indices = range(len(self))

        data = bl_utils.merge_dicts(
            {
                self.column_names.name: self.name,
                self.column_names.index: list(indices),
            },
            self.data.categories,
            latter_precedence=False,
        )

        available_time_info = map(
            bl_utils.is_not(None),
            (self.data.fps, self.data.ref_index, self.data.ref_elapsed_time),
        )
        if all(available_time_info):
            ref_index = self.data.ref_index
            ref_elapsed_time = pd.to_timedelta(
                self.data.ref_elapsed_time, unit='s'
            )
            delta = pd.to_timedelta(1 / self.data.fps, unit='s')
            elapsed_time_list = [
                ref_elapsed_time + delta * (index - ref_index)
                for index in indices
            ]

            data[self.column_names.elapsed_time] = elapsed_time_list
        elif enforce_time:
            raise ValueError(
                'there is not enough time info in video data'
                ' (set *enforce_time*=False to suppress this error).'
            )

        if self.column_names.path is not None:
            paths = sorted(self.glob_frames(), key=operator.attrgetter('stem'))
            data[self.column_names.path] = paths

        df = pd.DataFrame(data)
        df = self.convert_dataframe_type(
            df, categories_as_int=categories_as_int
        )

        if inplace:
            self.df = df
        return df

    def sync_time_series(
        self, source_df: pd.DataFrame, inplace: bool = True
    ) -> pd.DataFrame:
        df = self.make_dataframe(
            recalculate=False, enforce_time=True, inplace=inplace
        )

        df = sync_dataframes(
            source_df=source_df,
            dest_df=df,
            dest_time_column=self.column_names.elapsed_time,
        )

        if inplace:
            self.df = df

        return df

    def iterdata_from_dataframe(
        self, *, select_columns: Optional[Union[str, List[str]]] = None
    ) -> Iterable[Tuple[np.ndarray, Any]]:
        df = self.make_dataframe(recalculate=False)
        indices = df[self.column_names.index]

        data = df
        if select_columns is not None:
            data = data[select_columns]
            if not isinstance(select_columns, str):
                data = data.to_dict(orient='records')

        return zip(map(self.__getitem__, indices), data)

    def load_df(
        self,
        path: Optional[PathLike] = None,
        columns: Optional[Iterable[str]] = None,
        overwrite: bool = False,
        missing_ok: bool = False,
        inplace: bool = True,
    ) -> Optional[pd.DataFrame]:
        if self.df_path is None and path is None:
            raise ValueError(
                '*df_path* is not defined yet, so *path* must be given as argument.'
            )

        if not overwrite and self.df is not None:
            return self.df

        if path is None:
            path = self.df_path
        else:
            self.df_path = ensure_resolved(path)

        if missing_ok and not self.df_path.is_file():
            return None

        if columns is None:
            df = pd.read_csv(self.df_path, skipinitialspace=True)
        else:
            df = pd.read_csv(
                self.df_path, skipinitialspace=True, usecols=tuple(columns)
            )

        if inplace:
            self.df = df
        return df

    def save_df(
        self, path: Optional[PathLike] = None, overwrite: bool = False
    ) -> None:
        if self.df_path is None:
            raise ValueError(
                '*df_path* is not defined yet, so *path* must be given as argument.'
            )

        if path is None:
            path = self.df_path
        path = bl_utils.ensure_parent(path)

        if overwrite or not path.is_file():
            self.df.to_csv(path, index=False)

    def move_df(
        self,
        path: Union[str, bl_utils.PathLike],
        renaming: bool = False,
        erase_old: bool = False,
    ) -> None:
        if self.df_path is None and (erase_old or renaming):
            raise ValueError(
                '*erase_old* and *renaming* cannot be used '
                'if *df_path* is not defined yet.'
            )

        if erase_old:
            old_path = self.df_path

        if renaming:
            self.df_path = self.df_path.with_name(path)
        else:
            self.df_path = ensure_resolved(path)

        if erase_old and old_path.is_file():
            old_path.unlink()
        # if erase: # Python 3.8 only
        #     old_path.unlink(missing_ok=True)

    def frames_to_tensor(self, overwrite: bool = False) -> tf.data.Dataset:
        if self.frames_tensor_path is None and overwrite:
            raise ValueError(
                '`frames_tensor_path` is not defined yet. '
                'Please define it or pass `overwrite=False`.'
            )

        if (
            self.frames_tensor_path is not None
            and self.frames_tensor_path.is_file()
            and not overwrite
        ):
            return load_dataset(self.frames_tensor_path)

        return tf.data.Dataset.from_generator(lambda: self, tf.float32)

    def as_pairs(
        self,
        *,
        image_preprocessor: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = None,
        select_columns: Optional[Union[str, List[str]]] = None,
    ) -> Slicerator[Tuple[np.ndarray, Dict[str, Any]]]:
        df = self.make_dataframe(recalculate=False)
        df = self.convert_dataframe_type(df)
        df = df.sort_values(by=self.column_names.index)

        if select_columns is not None:
            df = df[select_columns]

        targets = df.to_dict('records')

        if image_preprocessor is not None:

            def get_item(i: int) -> Tuple[np.ndarray, Dict[str, Any]]:
                return image_preprocessor(self[i]), targets[i]

        else:

            def get_item(i: int) -> Tuple[np.ndarray, Dict[str, Any]]:
                return self[i], targets[i]

        return Slicerator.from_func(get_item, length=len(self))

    def as_tf_dataset(
        self,
        *,
        select_columns: Optional[Union[str, List[str]]] = None,
        inplace: bool = False,
    ) -> tf.data.Dataset:
        # See <https://www.tensorflow.org/tutorials/load_data/pandas_dataframe>

        pairs = self.as_pairs(select_columns=select_columns)
        type_spec = auto_spec(pairs[0])

        ds = tf.data.Dataset.from_generator(
            lambda: pairs, output_signature=type_spec
        )

        if inplace:
            self.ds = ds

        return ds
