from __future__ import annotations

import json as _json
from typing import Mapping, Optional

from boiling_learning.io.storage import dataclass
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.experiment_video_dataset import ExperimentVideoDataset
from boiling_learning.utils.dataclasses import dataclass_from_mapping
from boiling_learning.utils.pathutils import PathLike, resolve


class Case(ExperimentVideoDataset):
    @dataclass(frozen=True)
    class VideoDataKeys(ExperimentVideo.VideoDataKeys):
        name: str = 'name'
        ignore: str = 'ignore'

    def __init__(
        self,
        path: PathLike,
        name: Optional[str] = None,
        dataframes_dir_name: str = 'dataframes',
        videos_dir_name: str = 'videos',
        video_suffix: str = '.mp4',
        column_names: ExperimentVideo.DataFrameColumnNames = (
            ExperimentVideo.DataFrameColumnNames()
        ),
        column_types: ExperimentVideo.DataFrameColumnTypes = (
            ExperimentVideo.DataFrameColumnTypes()
        ),
        video_data_path: Optional[PathLike] = None,
    ) -> None:
        if not video_suffix.startswith('.'):
            raise ValueError('argument *video_suffix* must start with a dot \'.\'')

        self.path = resolve(path, dir=True)
        super().__init__(name=name or self.path.name)

        self.dataframes_dir = resolve(self.path / dataframes_dir_name, dir=True)
        self.videos_dir = resolve(self.path / videos_dir_name, dir=True)

        self.update(
            ExperimentVideo(
                video_path=video_path,
                df_dir=self.dataframes_dir,
                df_suffix='.csv',
                column_names=column_names,
                column_types=column_types,
            )
            for video_path in self.videos_dir.rglob(f'*{video_suffix}')
        )

        self.video_data_path = video_data_path or self.path / 'data.json'

    def set_video_data_from_file(
        self,
        data_path: PathLike,
        *,
        remove_absent: bool = False,
        keys: VideoDataKeys = VideoDataKeys(),
    ) -> None:
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
            name: dataclass_from_mapping(data, ExperimentVideo.VideoData, key_map=keys)
            for name, data in video_data.items()
        }

        self._set_video_data(video_data, remove_absent=remove_absent)

    def _set_video_data(
        self,
        video_data: Mapping[str, ExperimentVideo.VideoData],
        *,
        remove_absent: bool = False,
    ) -> None:
        video_data_keys = frozenset(video_data.keys())
        self_keys = frozenset(self.keys())
        for name in self_keys & video_data_keys:
            self[name].data = video_data[name]

        if remove_absent:
            for name in self_keys - video_data_keys:
                del self[name]

    def convert_videos(
        self, new_suffix: str, new_videos_dir: PathLike, overwrite: bool = False
    ) -> None:
        if not new_suffix.startswith('.'):
            raise ValueError('new_suffix is expected to start with a dot (\'.\')')

        new_videos_dir = resolve(new_videos_dir, root=self.path, dir=True)
        for experiment_video in self:
            tail = experiment_video.path.relative_to(self.videos_dir)
            dest_path = (new_videos_dir / tail).with_suffix(new_suffix)
            experiment_video.convert_video(dest_path, overwrite=overwrite)
        self.videos_dir = new_videos_dir
