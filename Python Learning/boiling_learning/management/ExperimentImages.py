from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional
)
import operator

import pandas as pd

import boiling_learning.utils as bl_utils
import boiling_learning.io as bl_io
import boiling_learning.preprocessing as bl_preprocessing
from boiling_learning.utils import PathType

class ExperimentImages:
    def __init__(
        self,
        path: Optional[PathType] = None,
        root_path: Optional[PathType] = None,
        name: Optional[str] = None,
        videos_path: Optional[PathType] = None,
        videos_dir_name: str = 'videos',
        audios_dir_name: str = 'audios',
        frames_dir_name: str = 'frames',
        video_suffix: str = '.mp4',
        audio_suffix: str = '.m4a',
        frame_suffix: str = '.png',
        video_data_path: Optional[PathType] = None
    ):
        if path is not None:
            self.path = bl_utils.ensure_dir(path)
            self.name = self.path.name
        elif root_path is not None and name is not None:
            self.path = bl_utils.ensure_dir(name, root=root_path)
            self.name = name
        else:
            raise ValueError('either path or both root_path and name should be given.')
        
        if videos_path is None:
            self.videos_dir_name = videos_dir_name
            self.videos_path = self.path / self.videos_dir_name
        else:
            videos_path = Path(videos_path)
            self.videos_dir_name = videos_path.name
            self.videos_path = videos_path
            
        self.audios_dir_name = audios_dir_name
        self.audios_path = self.path / self.audios_dir_name
        
        self.frames_dir_name = frames_dir_name
        self.frames_path = self.path / self.frames_dir_name
        
        if not all(suffix.startswith('.') for suffix in (video_suffix, audio_suffix, frame_suffix)):
            raise ValueError('parameters video_suffix, audio_suffix and frame_suffix are expected to start with a dot (\'.\'), and I don\'t know what do if this is not the case. Sorry, raising.')
        
        self.video_suffix = video_suffix
        self.audio_suffix = audio_suffix
        self.frame_suffix = frame_suffix
        
        if video_data_path is None:
            video_data_path = self.videos_path / 'data.json'
        self.video_data_path = video_data_path

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
    def subcases(self) -> Iterable[Path]:
        return map(
            operator.attrgetter('stem'),
            self.video_paths
        )

    @property
    def subcase_frames_dict(self) -> Dict[str, Iterable[Path]]:
        # TODO: use caching

        return {
            subcase: (self.frames_path / subcase).rglob('*' + self.frame_suffix)
            for subcase in self.subcases
        }

    @property
    def purged_subcase_frames_dict(self) -> Dict[str, Iterable[Path]]:
        return {
            subcase: value
            for subcase, value in self.subcase_frames_dict.items()
            if subcase in self.video_data(purge=True)
        }

    def extract_audios(
            self,
            overwrite: bool = False,
            verbose: bool = False
    ) -> None:
        for subcase, video_path in zip(self.subcases, self.video_paths):
            audio_path = (self.audios_path / subcase).with_suffix(self.audio_suffix)

            bl_utils.video.extract_audio(
                video_path,
                audio_path,
                overwrite=overwrite,
                verbose=verbose
            )

    def extract_frames(
            self,
            overwrite: bool = False,
            verbose: bool = False,
            chunk_sizes: Optional[List[int]] = None
    ) -> None:
        for subcase, video_path in zip(self.subcases, self.video_paths):

            if chunk_sizes is None:
                filename_pattern = f'{subcase}_frame%d{self.frame_suffix}'
            else:
                filename_pattern = chunked_filename_pattern(
                    chunk_sizes=chunk_sizes,
                    chunk_name='{min_index}-{max_index}',
                    filename=f'{subcase}_frame{{index}}{self.frame_suffix}'
                )

            bl_preprocessing.video.extract_frames(
                video_path,
                outputdir=self.frames_path / subcase,
                filename_pattern=filename_pattern,
                frame_suffix=self.frames_suffix,
                verbose=verbose,
                fast_frames_count=True,
                overwrite=overwrite
            )

            bl_utils.print_verbose(verbose, 'Done.')

    def video_data(self, purge: bool = True) -> Dict[str, Any]:
        loaded_data = bl_io.load_json(self.video_data_path)

        if purge:
            loaded_data = (
                item 
                for item in loaded_data 
                if not item.get('ignore', False)
            )

        return {
            item['name']: item['data']
            for item in loaded_data
        }

    def as_dataframe(
            self,
            path_column: str = 'path',
            subcase_column: str = 'subcase',
            predefined_column_types: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        data = self.video_data(purge=True)

        if predefined_column_types is None:
            predefined_column_types = {subcase_column: 'category'}

        # for subcase, paths in self.subcase_frames_dict.items():
        #     if subcase not in data:
        #         continue
        #     subcase_data = data[subcase]

        #     if bl_utils.has_duplicates((path_column, subcase_column) + tuple(subcase_data.keys())):
        #         raise ValueError(f'incompatibility between path column named "{path_column}", subcase column named "{subcase_column}" and data keys {list(subcase_data.keys())} from subcase "{subcase}". Make sure that path_column and subcase_column are not data keys and are different from each other.')

        #     dataframe = dataframe.append(
        #         pd.DataFrame(
        #             bl_utils.merge_dicts(
        #                 {
        #                     path_column: list(paths),
        #                     subcase_column: subcase
        #                 },
        #                 subcase_data
        #             )
        #         ),
        #         ignore_index=True
        #     ).astype(predefined_column_types)

        # # -- faster implementation: --

        return pd.concat(
            (
                pd.DataFrame(
                    bl_utils.merge_dicts(
                        {
                            path_column: list(paths),
                            subcase_column: subcase
                        },
                        data[subcase]
                    )
                )
                for subcase, paths in self.subcase_frames_dict.items()
                if subcase in data
            ),
            ignore_index=True
        ).astype(predefined_column_types)

    def as_dataset(
            self,
            df_path: Optional[PathType] = None,
            path_column: str = 'path',
            target_column: str = 'target',
            set_column: Optional[str] = None,
            train_key: str = 'train',
            val_key: str = 'val',
            test_key: str = 'test',
            subcase_column: str = 'subcase',
            predefined_column_types: Optional[Dict[str, Any]] = None,
            exist_load: bool = False
    ) -> bl_preprocessing.ImageDataset:
        
        if df_path is None:
            df_path = self.path / 'datasets' / 'dataset.csv'
            
        if exist_load and df_path.is_file():
            df = None
        else:
            df = self.as_dataframe(
                path_column=path_column,
                subcase_column=subcase_column,
                predefined_column_types=predefined_column_types
            )
            
        return bl_preprocessing.ImageDataset(
            df_path=df_path,
            path_column=path_column,
            target_column=target_column,
            set_column=set_column,
            train_key=train_key,
            val_key=val_key,
            test_key=test_key,
            df=df,
            exist_load=exist_load
        )