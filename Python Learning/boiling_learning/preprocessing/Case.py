import boiling_learning.utils as bl_utils
from boiling_learning.utils import (PathType, VerboseType)
from boiling_learning.preprocessing.ExperimentVideo import ExperimentVideo
from boiling_learning.preprocessing.ImageDataset import ImageDataset


class Case(ImageDataset):
    DataFrameColumnNames = ImageDataset.DataFrameColumnNames
    DataFrameColumnTypes = ImageDataset.DataFrameColumnTypes

    def __init__(
            self,
            path: PathType,
            df_name: str = 'dataset.csv',
            videos_dir_name: str = 'videos',
            audios_dir_name: str = 'audios',
            frames_dir_name: str = 'frames',
            video_suffix: str = '.mp4',
            audio_suffix: str = '.m4a',
            frames_suffix: str = '.png',
            column_names: DataFrameColumnNames = DataFrameColumnNames(),
            column_types: DataFrameColumnTypes = DataFrameColumnTypes()
    ):
        if not video_suffix.startswith('.'):
            raise ValueError(
                'argument *video_suffix* must start with a dot \'.\'')

        if not audio_suffix.startswith('.'):
            raise ValueError(
                'argument *audio_suffix* must start with a dot \'.\'')

        if not frames_suffix.startswith('.'):
            raise ValueError(
                'argument *frames_suffix* must start with a dot \'.\'')

        self.path = bl_utils.ensure_dir(path)
        df_path = self.path / df_name
        self.videos_dir = bl_utils.ensure_dir(self.path / videos_dir_name)
        self.audios_dir = bl_utils.ensure_dir(self.path / audios_dir_name)
        self.frames_dir = bl_utils.ensure_dir(self.path / frames_dir_name)

        super().__init__(
            column_names=column_names,
            column_types=column_types,
            df_path=df_path
        )

        for video_path in self.videos_dir.rglob('*' + video_suffix):
            self.add(
                ExperimentVideo(
                    video_path=video_path,
                    frames_dir=self.frames_dir / video_path.stem,
                    frames_suffix=frames_suffix,
                    audio_dir=self.audios_dir,
                    audio_suffix=audio_suffix,
                    column_names=column_names,
                    column_types=column_types
                )
            )

    def convert_videos(
            self,
            new_suffix: str,
            new_videos_dir: PathType,
            overwrite: bool = False,
            verbose: VerboseType = False
    ) -> None:
        if not new_suffix.startswith('.'):
            raise ValueError(
                'new_suffix is expected to start with a dot (\'.\')')

        new_videos_dir = bl_utils.ensure_dir(new_videos_dir, root=self.path)
        for element_video in self.values():
            tail = element_video.video_path.relative_to(self.videos_dir)
            dest_path = (new_videos_dir / tail).with_suffix(new_suffix)
            element_video.convert_video(
                dest_path,
                overwrite=overwrite,
                verbose=verbose
            )
        self.videos_dir = new_videos_dir

#     def as_dataframe(
#             self,
#             video_data: Optional[Mapping[str, Any]] = None,
#             video_data_keys: VideoDataKeys = VideoDataKeys(),
#             column_names: ColumnNames = ColumnNames(),
#             predefined_column_types: Optional[Dict[str, Any]] = None
#     ) -> pd.DataFrame:
#         if video_data is not None:
#             self.video_data = video_data

#         def _make_dict(subcase, paths):
#             subcase_data = self.video_data[subcase]

#             paths = list(paths)

#             id_dict = {
#                 column_names.path: paths,
#                 column_names.subcase: subcase
#             }

#             constant_data_dict = subcase_data[video_data_keys.data]

#             if all(map(
#                     subcase_data.__contains__,
#                     (video_data_keys.fps,
#                      video_data_keys.ref_image,
#                      video_data_keys.ref_instant)
#             )):
#                 fps = subcase_data[video_data_keys.fps]
#                 ref_image = subcase_data[video_data_keys.ref_image]
#                 ref_instant = datetime.datetime(
#                     subcase_data[video_data_keys.ref_instant])

#                 image_names = [path.name for path in paths]
#                 ref_index = image_names.find(ref_image)
#                 delta = datetime.timedelta(seconds=1/fps)
#                 time_instants = [
#                     ref_instant + delta*(index - ref_index)
#                     for index in bl_utils.indexify(paths)
#                 ]

#                 time_dict = {
#                     column_names.time_instant: time_instants
#                 }
#             else:
#                 time_dict = {}

#             return bl_utils.merge_dicts(
#                 id_dict,
#                 time_dict,
#                 constant_data_dict,
#                 latter_precedence=False
#             )

#         dfs = (
#             pd.DataFrame(
#                 _make_dict(subcase, paths)
#             )
#             for subcase, paths in self.subcase_frames_dict.items()
#             if subcase in self.video_data
#         )

#         df = pd.concat(
#             dfs,
#             ignore_index=True
#         )

#         if predefined_column_types is None:
#             predefined_column_types = {
#                 column_names.subcase: 'category',
#                 column_names.time_instant: 'datetime'
#             }
#         df_columns = set(df.columns)
#         predefined_column_types = {
#             key: value
#             for key, value in predefined_column_types.items() if key in df_columns
#         }

#         return df.astype(predefined_column_types)
