from typing import Optional

from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.image_datasets import ImageDataset
from boiling_learning.utils import PathLike, resolve


class Case(ImageDataset):
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
        data_path: Optional[PathLike] = None,
        remove_absent: bool = False,
        keys: ImageDataset.VideoDataKeys = ImageDataset.VideoDataKeys(),
    ) -> None:
        super().set_video_data_from_file(
            data_path or self.video_data_path,
            keys=keys,
            remove_absent=remove_absent,
        )

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
