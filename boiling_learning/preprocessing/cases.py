from __future__ import annotations

import json as _json
from dataclasses import asdict
from typing import Optional

import modin.pandas as pd
from loguru import logger

from boiling_learning.dataclasses import dataclass_from_mapping
from boiling_learning.io.storage import dataclass
from boiling_learning.preprocessing.experiment_video import ExperimentVideo, VideoData
from boiling_learning.preprocessing.experiment_video_dataset import ExperimentVideoDataset
from boiling_learning.utils.pathutils import PathLike, resolve


class Case(ExperimentVideoDataset):
    @dataclass(frozen=True)
    class VideoDataKeys(ExperimentVideo.VideoDataKeys):
        name: str = 'name'
        ignore: str = 'ignore'

    def __init__(
        self,
        path: PathLike,
        dataframes_dir_name: str = 'dataframes',
        videos_dir_name: str = 'videos',
        video_suffix: str = '.mp4',
        video_data_path: Optional[PathLike] = None,
        experimental_data_path: Optional[PathLike] = None,
    ) -> None:
        if not video_suffix.startswith('.'):
            raise ValueError('argument `video_suffix` must start with a dot \'.\'')

        self.path = resolve(path, dir=True)
        self.dataframes_dir = resolve(self.path / dataframes_dir_name, dir=True)
        self.videos_dir = resolve(self.path / videos_dir_name, dir=True)

        super().__init__(
            ExperimentVideo(
                video_path=video_path,
                df_path=self.dataframes_dir / f'{video_path.stem}.csv',
            )
            for video_path in self.videos_dir.rglob(f'*{video_suffix}')
        )

        self.video_data_path = resolve(video_data_path or self.path / 'data.json')
        self.experimental_data_path = resolve(experimental_data_path or self.path / 'data.csv')
        self.df: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return self.path.name

    def set_video_data_from_file(
        self,
        *,
        keys: VideoDataKeys = VideoDataKeys(),
    ) -> ExperimentVideoDataset:
        data_path = resolve(self.video_data_path)
        video_data = _json.loads(data_path.read_text(encoding='utf8'))

        if isinstance(video_data, list):
            video_data = {
                item.pop(keys.name): item
                for item in video_data
                if not item.pop(keys.ignore, False)
            }
        elif isinstance(video_data, dict):
            video_data = {
                key: value
                for key, value in video_data.items()
                if not value.pop(keys.ignore, False)
            }
        else:
            raise RuntimeError(f'could not load video data from {data_path}. Got {video_data!r}.')

        video_data = {
            name: dataclass_from_mapping(
                data,
                VideoData,
                key_map=asdict(keys),
            )
            for name, data in video_data.items()
        }

        return ExperimentVideoDataset(
            ev.with_data(video_data[ev.name]) for ev in self if ev.name in video_data
        )

    def get_experimental_data(self) -> pd.DataFrame:
        if self.df is not None:
            return self.df

        return (
            pd.read_csv(self.experimental_data_path)
            .drop(columns='Time instant')
            .astype({'Elapsed time': 'float64'})
            .set_index('Elapsed time')
        )

    def convert_videos(
        self,
        new_suffix: str,
        new_videos_dirname: str,
        overwrite: bool = False,
    ) -> Case:
        logger.info(f'Converting videos for case {self.name}...')

        if not new_suffix.startswith('.'):
            raise ValueError('new_suffix is expected to start with a dot (\'.\')')

        new_videos_dir = resolve(self.path / new_videos_dirname, dir=True)
        for experiment_video in self:
            tail = experiment_video.path.relative_to(self.videos_dir)
            dest_path = (new_videos_dir / tail).with_suffix(new_suffix)
            experiment_video.convert_video(dest_path, overwrite=overwrite)

        logger.info(f'Successfully converted videos for case {self.name}...')

        return Case(
            path=self.path,
            dataframes_dir_name=self.dataframes_dir.name,
            videos_dir_name=new_videos_dirname,
            video_suffix=new_suffix,
            video_data_path=self.video_data_path,
        )
