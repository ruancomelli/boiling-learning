from dataclasses import dataclass
import datetime
import operator
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Union
)

import pandas as pd

import boiling_learning.utils as bl_utils
from boiling_learning.utils import PathType
from boiling_learning.preprocessing import video as bl_preprocessing_video

VideoDataType = Dict[str, Any]


def read_video_data(
    data_path: PathType,
    purge: bool = False,
    name_key: str = 'name',
    ignore_key: str = 'ignore'
) -> VideoDataType:
    return {
        item.pop(name_key): item
        for item in bl_io.load_json(data_path)
        if not (purge and item.get(ignore_key, False))
    }


class ImageData:
    @dataclass(frozen=True)
    class ColumnNames:
        path: str = 'path'
        subcase: str = 'subcase'
        time_instant: str = 'time_instant'

    @dataclass(frozen=True)
    class VideoDataKeys:
        data: str = 'data'
        fps: str = 'fps'
        ref_image: str = 'ref_image'
        ref_instant: str = 'ref_instant'

    @dataclass(frozen=True)
    class DirNames:
        videos: str = 'videos'
        audios: str = 'audios'
        frames: str = 'frames'

    @dataclass(frozen=True)
    class Suffixes:
        videos: str = '.mp4'
        audios: str = '.m4a'
        frames: str = '.png'

    def __init__(
        self,
        path: Optional[PathType] = None,
        root_path: Optional[PathType] = None,
        name: Optional[str] = None,
        videos_path: Optional[PathType] = None,
        dir_names: DirNames = DirNames(),
        suffixes: Suffixes = Suffixes(),
        video_data: Optional[VideoDataType] = None
    ):
        self.path: Path
        self.name: str
        self.videos_path: Path
        self.audios_path: Path
        self.frames_path: Path
        self._video_suffix: str
        self._audio_suffix: str
        self._frame_suffix: str
        self.video_data: VideoDataType

        if path is not None:
            self.path = bl_utils.ensure_dir(path)
            self.name = self.path.name
        elif root_path is not None and name is not None:
            self.path = bl_utils.ensure_dir(name, root=root_path)
            self.name = name
        else:
            raise ValueError(
                'either path or both root_path and name should be given.')

        if videos_path is None:
            self.videos_path = self.path / dir_names.videos
        else:
            self.videos_path = bl_utils.ensure_resolved(videos_path)

        audios_dir_name = dir_names.audios
        self.audios_path = self.path / audios_dir_name

        frames_dir_name = dir_names.frames
        self.frames_path = self.path / frames_dir_name

        self.video_suffix = suffixes.videos
        self.audio_suffix = suffixes.audios
        self.frame_suffix = suffixes.frames

        if video_data is not None:
            self.video_data = video_data

    @property
    def video_suffix(self) -> str:
        return self._video_suffix

    @video_suffix.setter
    def video_suffix(self, new_suffix: str) -> None:
        if not new_suffix.startswith('.'):
            raise ValueError(
                'video_suffix is expected to start with a dot (\'.\'),'
                'and I don\'t know what do if this is not the case.'
                'Sorry, raising.')

        self._video_suffix = new_suffix

    @property
    def audio_suffix(self) -> str:
        return self._audio_suffix

    @audio_suffix.setter
    def audio_suffix(self, new_suffix: str) -> None:
        if not new_suffix.startswith('.'):
            raise ValueError(
                'audio_suffix is expected to start with a dot (\'.\'),'
                'and I don\'t know what do if this is not the case.'
                'Sorry, raising.')

        self._audio_suffix = new_suffix

    @property
    def frame_suffix(self) -> str:
        return self._frame_suffix

    @frame_suffix.setter
    def frame_suffix(self, new_suffix: str) -> None:
        if not new_suffix.startswith('.'):
            raise ValueError(
                'frame_suffix is expected to start with a dot (\'.\'),'
                'and I don\'t know what do if this is not the case.'
                'Sorry, raising.')

        self._frame_suffix = new_suffix

    @property
    def video_data(self) -> VideoDataType:
        if self._video_data is None:
            raise ValueError(
                'invalid access to video_data before setting its value.')
        return self._video_data

    @video_data.setter
    def video_data(self, video_data: VideoDataType) -> None:
        self._video_data = video_data

    @property
    def video_paths(self) -> Iterable[Path]:
        return self.videos_path.rglob('*' + self.video_suffix)

    @property
    def audio_paths(self) -> Iterable[Path]:
        return self.audios_path.rglob('*' + self.audio_suffix)

    @property
    def frame_paths(self) -> Iterable[Path]:
        return self.frames_path.rglob('*' + self.frame_suffix)

    @property
    def subcases(self) -> Iterable[str]:
        return map(
            operator.attrgetter('stem'),
            self.video_paths
        )

    def subcase_frames_path(self, subcase_name: str) -> Path:
        return self.frames_path / subcase_name

    @property
    def subcase_frames_dict(self) -> Dict[str, Iterable[Path]]:
        # TODO: use caching

        return {
            subcase: self.subcase_frames_path(subcase).rglob('*' + self.frame_suffix)
            for subcase in self.subcases
        }

    def convert_videos(
            self,
            new_suffix: str,
            new_videos_path: PathType,
            overwrite: bool = False,
            verbose: Union[bool, int] = False
    ) -> None:
        if not new_suffix.startswith('.'):
            raise ValueError(
                'new_suffix is expected to start with a dot (\'.\')')

        new_videos_path = bl_utils.ensure_dir(new_videos_path, root=self.path)
        for video_path in self.video_paths:
            tail = video_path.relative_to(
                self.videos_path).with_suffix(new_suffix)
            dest_path = new_videos_path / tail
            bl_preprocessing_video.convert_video(
                video_path, dest_path, verbose=verbose, overwrite=overwrite)
        self.videos_path = new_videos_path
        self.video_suffix = new_suffix

    def extract_audios(
            self,
            overwrite: bool = False,
            verbose: bool = False
    ) -> None:
        for subcase, video_path in zip(self.subcases, self.video_paths):
            audio_path = (self.audios_path /
                          subcase).with_suffix(self.audio_suffix)

            bl_preprocessing_video.extract_audio(
                video_path,
                audio_path,
                overwrite=overwrite,
                verbose=verbose
            )

    def extract_frames(
            self,
            overwrite: bool = False,
            verbose: Union[bool, int] = False,
            chunk_sizes: Optional[List[int]] = None,
            iterate: bool = True
    ) -> None:
        for subcase, video_path in zip(self.subcases, self.video_paths):
            if chunk_sizes is None:
                filename_pattern = f'{subcase}_frame%d{self.frame_suffix}'
            else:
                filename_pattern = bl_preprocessing_video.chunked_filename_pattern(
                    chunk_sizes=chunk_sizes,
                    chunk_name='{min_index}-{max_index}',
                    filename=f'{subcase}_frame{{index}}{self.frame_suffix}'
                )

            bl_preprocessing_video.extract_frames(
                video_path,
                outputdir=self.frames_path / subcase,
                filename_pattern=filename_pattern,
                frame_suffix=self.frame_suffix,
                verbose=verbose,
                fast_frames_count=True,
                overwrite=overwrite,
                iterate=iterate
            )

            bl_utils.print_verbose(verbose, 'Done.')

    def as_dataframe(
            self,
            video_data: Optional[Mapping[str, Any]] = None,
            video_data_keys: VideoDataKeys = VideoDataKeys(),
            column_names: ColumnNames = ColumnNames(),
            predefined_column_types: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        if video_data is not None:
            self.video_data = video_data

        def _make_dict(subcase, paths):
            subcase_data = self.video_data[subcase]

            paths = list(paths)

            id_dict = {
                column_names.path: paths,
                column_names.subcase: subcase
            }

            constant_data_dict = subcase_data[video_data_keys.data]

            if all(map(
                    bl_utils.contained(subcase_data),
                    (video_data_keys.fps,
                     video_data_keys.ref_image,
                     video_data_keys.ref_instant)
            )):
                fps = subcase_data[video_data_keys.fps]
                ref_image = subcase_data[video_data_keys.ref_image]
                ref_instant = datetime.datetime(
                    subcase_data[video_data_keys.ref_instant])

                image_names = [path.name for path in paths]
                ref_index = image_names.find(ref_image)
                delta = datetime.timedelta(seconds=1/fps)
                time_instants = [
                    ref_instant + delta*(index - ref_index)
                    for index in bl_utils.indexify(paths)
                ]

                time_dict = {
                    column_names.time_instant: time_instants
                }
            else:
                time_dict = {}

            return bl_utils.merge_dicts(
                id_dict,
                time_dict,
                constant_data_dict,
                latter_precedence=False
            )

        dfs = (
            pd.DataFrame(
                _make_dict(subcase, paths)
            )
            for subcase, paths in self.subcase_frames_dict.items()
            if subcase in self.video_data
        )

        df = pd.concat(
            dfs,
            ignore_index=True
        )

        if predefined_column_types is None:
            predefined_column_types = {
                column_names.subcase: 'category',
                column_names.time_instant: 'datetime'
            }
        df_columns = set(df.columns)
        predefined_column_types = {
            key: value
            for key, value in predefined_column_types.items()
            if key in df_columns
        }

        return df.astype(predefined_column_types)
