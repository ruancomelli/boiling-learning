from typing import Iterable, Optional

import modin.pandas as pd

from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.image_datasets import ImageDataset
from boiling_learning.utils.utils import PathLike, VerboseType, resolve


class Case(ImageDataset):
    VideoDataKeys = ImageDataset.VideoDataKeys
    DataFrameColumnNames = ImageDataset.DataFrameColumnNames
    DataFrameColumnTypes = ImageDataset.DataFrameColumnTypes

    def __init__(
        self,
        path: PathLike,
        name: Optional[str] = None,
        df_name: str = 'dataset.csv',
        dataframes_dir_name: str = 'dataframes',
        videos_dir_name: str = 'videos',
        audios_dir_name: str = 'audios',
        frames_dir_name: str = 'frames',
        frames_tensor_dir_name: str = 'frame_tensors',
        video_suffix: str = '.mp4',
        audio_suffix: str = '.m4a',
        frames_suffix: str = '.png',
        column_names: DataFrameColumnNames = DataFrameColumnNames(),
        column_types: DataFrameColumnTypes = DataFrameColumnTypes(),
        video_data_path: Optional[PathLike] = None,
        tags: Iterable[str] = (),
    ) -> None:
        if not video_suffix.startswith('.'):
            raise ValueError('argument *video_suffix* must start with a dot \'.\'')

        if not audio_suffix.startswith('.'):
            raise ValueError('argument *audio_suffix* must start with a dot \'.\'')

        if not frames_suffix.startswith('.'):
            raise ValueError('argument *frames_suffix* must start with a dot \'.\'')

        self.path = resolve(path, dir=True)

        if name is None:
            name = self.path.name

        df_path = self.path / df_name
        self.dataframes_dir = resolve(self.path / dataframes_dir_name, dir=True)
        self.videos_dir = resolve(self.path / videos_dir_name, dir=True)
        self.audios_dir = resolve(self.path / audios_dir_name, dir=True)
        self.frames_dir = resolve(self.path / frames_dir_name, dir=True)
        self.frames_tensor_dir = resolve(self.path / frames_tensor_dir_name, dir=True)

        super().__init__(
            name=name,
            column_names=column_names,
            column_types=column_types,
            df_path=df_path,
            tags=tags,
        )

        for video_path in self.videos_dir.rglob('*' + video_suffix):
            self.add(
                ExperimentVideo(
                    video_path=video_path,
                    frames_dir=self.frames_dir,
                    frames_suffix=frames_suffix,
                    frames_tensor_dir=self.frames_tensor_dir,
                    audio_dir=self.audios_dir,
                    audio_suffix=audio_suffix,
                    df_dir=self.dataframes_dir,
                    df_suffix='.csv',
                    column_names=column_names,
                    column_types=column_types,
                )
            )

        if video_data_path is None:
            video_data_path = self.path / 'data.json'
        self.video_data_path = video_data_path

    def set_video_data_from_file(
        self,
        video_data_path: Optional[PathLike] = None,
        purge: bool = False,
        keys: VideoDataKeys = VideoDataKeys(),
        remove_absent: bool = False,
    ) -> None:
        if video_data_path is None:
            video_data_path = self.video_data_path

        super().set_video_data_from_file(
            video_data_path,
            purge=purge,
            keys=keys,
            remove_absent=remove_absent,
        )

    def convert_videos(
        self,
        new_suffix: str,
        new_videos_dir: PathLike,
        overwrite: bool = False,
        verbose: VerboseType = False,
    ) -> None:
        if not new_suffix.startswith('.'):
            raise ValueError('new_suffix is expected to start with a dot (\'.\')')

        new_videos_dir = resolve(new_videos_dir, root=self.path, dir=True)
        for element_video in self.values():
            tail = element_video.path.relative_to(self.videos_dir)
            dest_path = (new_videos_dir / tail).with_suffix(new_suffix)
            element_video.convert_video(dest_path, overwrite=overwrite, verbose=verbose)
        self.videos_dir = new_videos_dir

    def sync_time_series(self, source_df: pd.DataFrame, inplace: bool = True) -> None:
        for experiment_video in self.values():
            experiment_video.sync_time_series(source_df, inplace=True)