from pathlib import Path

from more_itertools import flatten

from boiling_learning.preprocessing import ImageDataset

class Case:
    def __init__(
        self,
        path=None,
        root_path=None,
        name=None,
        videos_path=None,
        videos_dir_name='videos',
        audios_dir_name='audios',
        frames_dir_name='frames',
        video_format='mp4',
        audio_format='m4a',
        frame_format='png',
        video_data_path=None
    ):
        # TODO: either path or (root_path and name) should be given
        if path is None:
            self.root_path = Path(root_path)
            self.path = self.root_path / name
            self.name = name
        else:
            self.path = Path(path)
            self.name = self.path.name 
        
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
            
        self.video_format = video_format
        self.audio_format = audio_format
        self.frame_format = frame_format
        
        if video_data_path is None:
            video_data_path = self.videos_path / 'data.json'
        self.video_data_path = video_data_path

    @property
    def video_path_list(self):
        return list(self.glob_videos(glob_pattern=f'*.{self.video_format}'))
    
    @property
    def audio_path_list(self):
        return list(self.glob_audios(glob_pattern=f'*.{self.audio_format}'))
    
    @property
    def frame_path_list(self):
        return list(self.glob_frames(glob_pattern=f'**/*.{self.frame_format}'))
    
    def glob_videos(self, glob_pattern=None, glob_patterns=None):
        # TODO: use check_value_match
        
        if glob_patterns is not None:
            return flatten(self.glob_videos(glob_pattern=glob_pattern) for glob_pattern in glob_patterns)
        
        return self.videos_path.glob(glob_pattern)
    
    def glob_audios(self, glob_pattern=None, glob_patterns=None):
        # TODO: use check_value_match
        
        if glob_patterns is not None:
            return flatten(self.glob_audios(glob_pattern=glob_pattern) for glob_pattern in glob_patterns)
        
        return self.audios_path.glob(glob_pattern)
    
    def glob_frames(self, glob_pattern=None, glob_patterns=None):
        # TODO: use check_value_match
        
        if glob_patterns is not None:
            return flatten(self.glob_frames(glob_pattern=glob_pattern) for glob_pattern in glob_patterns)
        
        return self.frames_path.glob(glob_pattern)

    @property
    def subcase_list(self):
        # TODO: use caching
        
        return [video_path.stem for video_path in self.video_path_list]
        
    @property
    def subcase_frames_dict(self):
        # TODO: use caching
        
        return {
            subcase: self.glob_frames(glob_pattern=f'{subcase}/**/*.{self.frame_format}')
            for subcase in self.subcase_list
        }
    
    @property
    def purged_subcase_frames_dict(self):
        return {
            subcase: value
            for subcase, value in self.subcase_frames_dict.items()
            if subcase in self.video_data(purge=True)
        }
        
    def extract_audios(self, overwrite=False, verbose=False):
        for subcase, video_path in zip(self.subcase_list, self.video_path_list):
            audio_path = (self.audios_path / subcase).with_suffix(f'.{self.audio_format}')
            
            if verbose:
                bl.utils.print_header(f'Extracting audio for file \"{audio_path}\"')
            
            if overwrite or not audio_path.is_file():
                bl.utils.print_verbose(verbose, f'Audio does not exist. Extracting...', end=' ')
                bl.utils.video.extract_audio(video_path, audio_path)
                bl.utils.print_verbose(verbose, f'Done.')
            elif verbose:
                print(f'Audio already exists. Skipping proccess.')
                
        return self
                
    def extract_frames(self, overwrite=False, verbose=False, chunk_size=None):
        # TODO: check_value_match
        
        subcase_frames_dict = self.subcase_frames_dict
        
        for subcase, video_path in zip(self.subcase_list, self.video_path_list):
            video_frames_path = self.frames_path / subcase
            video_frames_path.mkdir(exist_ok=True, parents=True)
            
            if verbose:
                bl.utils.print_header(f'Extracting frames for file \"{str(video_frames_path)}\"')
            
            if not overwrite:
                n_extracted_frames = mit.ilen(subcase_frames_dict[subcase])
                n_frames = bl.utils.video.count_frames(video_path, fast=True)
                overwrite = n_extracted_frames != n_frames
                
            if overwrite:
                bl.utils.print_verbose(verbose, f'Frames do not exist or are incomplete. Extracting...', end=' ')
                    
                if chunk_size is None:
                    final_pattern = None
                else:
                    def final_pattern(index):
                        min_index = (index // chunk_size) * chunk_size
                        max_index = min_index + chunk_size - 1
                        return Path(f'from_{min_index}_to_{max_index}') / f'{subcase}_frame{index}.{self.frame_format}'
                    
                bl.utils.rmdir(video_frames_path, recursive=True, keep=True)
                bl.utils.video.extract_frames(
                    video_path,
                    video_frames_path,
                    filename_pattern=f'{subcase}_frame%05d.{self.frame_format}',
                    final_pattern=final_pattern,
                    make_parents=True,
                    verbose=False
                )
                bl.utils.print_verbose(verbose, f'Done.')
            else:
                bl.utils.print_verbose(verbose, f'Frames already extracted. Skipping proccess.')
                
        return self
    
    def video_data(self, purge=True):
        from json import load
        
        with self.video_data_path.open('r') as file:
            loaded_data = load(file)
        final_data = [item for item in loaded_data if not item.get('ignore', False)] if purge else loaded_data
        
        return {
            item['name']: item['data']
            for item in final_data
        }
    
    def as_dataframe(self, path_column='path', subcase_column='subcase', predefined_column_types=None):
        from itertools import repeat
        
        data = self.video_data(purge=True)
        
        if predefined_column_types is None:
            predefined_column_types = {subcase_column: 'category'}
        
        # for subcase, paths in self.subcase_frames_dict.items():
        #     if subcase not in data:
        #         continue
        #     subcase_data = data[subcase]
            
        #     if bl.utils.has_duplicates((path_column, subcase_column) + tuple(subcase_data.keys())):
        #         raise ValueError(f'incompatibility between path column named "{path_column}", subcase column named "{subcase_column}" and data keys {list(subcase_data.keys())} from subcase "{subcase}". Make sure that path_column and subcase_column are not data keys and are different from each other.')
            
        #     dataframe = dataframe.append(
        #         pd.DataFrame(
        #             bl.utils.merge_dicts(
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
                    bl.utils.merge_dicts(
                        {
                            path_column: list(paths),
                            subcase_column: subcase
                        },
                        data[subcase]
                    )
                )
                for subcase, paths in filter(lambda pair: pair[0] in data, self.subcase_frames_dict.items())
            ),
            ignore_index=True
        ).astype(predefined_column_types)
        
    def as_dataset(
        self,
        df_path=None,
        path_column='path',
        target_column='target',
        set_column=None,
        train_key='train',
        val_key='val',
        test_key='test',
        subcase_column='subcase',
        predefined_column_types=None,
        exist_load=False
    ):
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
            
        return ImageDataset(
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