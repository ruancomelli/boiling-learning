from pathlib import Path
import numpy as np
import pandas as pd
import random
import functools
import itertools
import more_itertools as mit

import skimage
import skimage.color

import boiling_learning as bl
from boiling_learning.management import Persistent, PersistentTransformer

_marker = object()

python_project_home_path = Path().absolute().resolve()
project_home_path = python_project_home_path.parent.resolve()

def load_persistent(path, auto_purge=False):
    from skimage import img_as_float, img_as_ubyte
    from skimage.io import imread, imsave
    
    def imread_as_float(path):
        try:
            return img_as_float(imread(path))
        except (SyntaxError, IOError):
            if auto_purge:
                path.unlink()
            raise
    
    def imsave_as_ubyte(path, img):
        return imsave(path, img_as_ubyte(img))
    
    return Persistent(
        path,
        checker=lambda x: x.is_file(),
        reader=imread_as_float,
        writer=imsave_as_ubyte,
        record_paths=True
    )

class TransformationPipeline:
    def __init__(self, *transformers):
        self._pipe = boiling_learning.utils.packed_functional.compose(*transformers)
        
    def transform(self, X, many=False, fetch=None, parallel=False):
        import os
        from functools import partial

        from more_itertools import consume
        
        should_fetch = callable(fetch) or bool(fetch)
        
        if many:
            transformer = partial(self.transform, many=False, fetch=fetch)
            
            if parallel:
                from pathos.multiprocessing import ProcessingPool
                # from multiprocessing import Pool
                
                if isinstance(parallel, bool):
                    processes = []
                else:
                    processes = [parallel]
                    
                with ProcessingPool(*processes) as pool:
                # with Pool(*processes) as pool:
                    bl.utils.print_header(f'Creating pool with ncpus={pool.ncpus} and nodes={pool.nodes}')
                    if should_fetch:
                        return pool.map(transformer, X)
                        # return pool.map(bl.utils.worker.apply_to_f, ((transformer, x, {}) for x in X))
                        # return pool.map(bl.utils.worker.apply_to_obj, ((self, 'transform', x, {}) for x in X))
                    else:
                        consume(pool.uimap(transformer, X))
                        # consume(pool.imap_unordered(transformer, X))
                        # consume(pool.imap_unordered(bl.utils.worker.apply_to_f, ((transformer, x, {}) for x in X)))
                        # consume(pool.imap_unordered(bl.utils.worker.apply_to_obj, ((self, 'transform', x, {}) for x in X)))
                        return self
                    
            else:
                mapper = map(transformer, X)
                if should_fetch:
                    return list(mapper)
                else:
                    consume(mapper)
                    return self
        else:
            if should_fetch:
                if callable(fetch):
                    return fetch(self._pipe(X))
                else:
                    return self._pipe(X)
            else:
                self._pipe(X)
                return self

class ArgGenerator:
    def __init__(self, f, generator, keyer=None, auto_generate=False):
        self._generator = generator
        self._store = dict()
        self._fun = bl.utils.packed_functional.packed(f)
        self._keyer = keyer
        self._auto_generate = auto_generate
        
    def _key(self, key):
        if self._keyer is None:
            return key
        else:
            return self._keyer(key)
        
    def __setitem__(self, key, value):
        self._store[self._key(key)] = value
        
    def __getitem__(self, key):
        return self._store[self._key(key)]
    
    def __contains__(self, key):
        return self._key(key) in self._store
    
    def generate(self, key):
        if key not in self:
            self[key] = self._generator(key)
        return self._store[key]
    
    def __call__(self, key=_marker):
        if key is _marker:
            return self._fun(bl.utils.packed_functional.pack())
            
        if self._auto_generate:
            self.generate(key)
        return self._fun(self[key])

@bl.utils.constant_callable
def random_coin():
    from random import choice
    
    return choice([False, True])

def auto_gen(arg_generator, key_index=0):
    def wrapped(*args, **kwargs):
        args = list(args)
        key = args.pop(key_index)
        return arg_generator(key)(*args, **kwargs)
    return wrapped

# an ImageDataset is a file CSV in df_path and the correspondent images. The file in df_path contains at least two columns. One of this columns contains file paths, and the other the targets for training, validation or test. This is intended for using flow_from_dataframe. There may be an optional column which specifies if that image belongs to the training, the validation or the test sets.
class ImageDataset:
    def __init__(
        self,
        df_path=None,
        path_column='path',
        target_column='target',
        set_column=None,
        train_key='train',
        val_key='val',
        test_key='test',
        df=None,
        exist_load=False
    ):
        from pathlib import Path
        
        self.df_path = Path(df_path)
        self.path_column = path_column
        self.target_column = target_column
        self.set_column = set_column
        self.train_key = train_key
        self.val_key = val_key
        self.test_key = test_key
        
        if exist_load and self.df_path.is_file():
            self.load()
        else:
            if df is None:
                df = pd.DataFrame()
            self.df = df
    
    def __repr__(self):
        return f'ImageDataset(df_path={self.df_path}, path_column={self.path_column}, target_column={self.target_column}, set_column={self.set_column}, train_key={self.train_key}, val_key={self.val_key}, test_key={self.test_key})'
    
    def __str__(self):
        return '\n'.join((
            f'ImageDataset at {self.df_path}',
            f'path_column={self.path_column}',
            f'target_column={self.target_column}',
            f'set_column={self.set_column}',
            f'train_key={self.train_key}',
            f'val_key={self.val_key}',
            f'test_key={self.test_key}',
            f'{self.df}'
        ))
        
    def load(self, path=None, cheap=False):
        if path is None:
            path = self.df_path
        else:
            self.df_path = path
        
        if cheap:
            if self.set_column is None:
                usecols = [self.path_column, self.target_column]
            else:
                usecols = [self.path_column, self.target_column, self.set_column]
            self.df = pd.read_csv(self.df_path, skipinitialspace=True, usecols=usecols)
        else:
            self.df = pd.read_csv(self.df_path, skipinitialspace=True)
            
        return self
            
    def save(self, path=None, exist_skip=False):
        if path is None:
            path = self.df_path
        path = Path(path)
        
        if not (exist_skip and path.is_file()):
            path.parent.mkdir(exist_ok=True, parents=True)
            self.df.to_csv(path)
        
        return self
    
    def move(self, path, renaming=False, erase=False, exist_skip=False):
        from pathlib import Path
        
        if erase:
            old_path = self.df_path
            
        if renaming:
            self.df_path = self.df_path.with_name(path)
        else:
            self.df_path = Path(path)
            
        self.save(exist_skip=exist_skip)
        
        if erase:
            old_path.unlink(missing_ok=True)
    
    @property
    def data(self):
        return self.df[[self.path_column, self.target_column]]
    
    @data.setter
    def data(self, other):
        self.df[[self.path_column, self.target_column]] = other
    
    @property
    def train_data(self):
        return self.data[self.df[self.set_column] == self.train_key]
    
    @train_data.setter
    def train_data(self, other):
        self.df.loc[self.df[self.set_column] == self.train_key, [self.path_column, self.target_column]] = other
    
    @property
    def val_data(self):
        return self.data[self.df[self.set_column] == self.val_key]
    
    @val_data.setter
    def val_data(self, other):
        self.df.loc[self.df[self.set_column] == self.val_key, [self.path_column, self.target_column]] = other
    
    @property
    def test_data(self):
        return self.data[self.df[self.set_column] == self.test_key]
    
    @test_data.setter
    def test_data(self, other):
        self.df.loc[self.df[self.set_column] == self.test_key, [self.path_column, self.target_column]] = other
    
    @property
    def paths(self):
        return self.df[self.path_column]
    
    @paths.setter
    def paths(self, other):
        self.df[self.path_column] = other
    
    @property
    def targets(self):
        return self.df[self.target_column]
    
    @targets.setter
    def targets(self, other):
        self.df[self.target_column] = other
    
    def modify_path(self, old_path, new_path, many=False):
        from pathlib import Path
        
        if many:
            old_to_new = dict(zip(map(Path, old_path), map(Path, new_path)))
            self.df[self.path_column] = self.df[self.path_column].apply(
                lambda y: old_to_new.get(Path(y), y)
            )
        else:
            self.df[self.path_column] = self.df[self.path_column].mask(
                lambda x: Path(x) == Path(old_path),
                new_path
            )
        
        return self
            
    def transform_images(self, transformer, **kwargs):
        # TODO: remove this code duplication
        # TODO: allow that only train data is modified
        # TODO: allow dataset augmentation, not only transformation
        
        return transformer.transform_images(self.paths, **kwargs)
    
    def append(self, other):
        self.append_dataframe(other.df)
        
        return self
    
    def append_dataframe(self, df):
        self.df.append(df, ignore_index=True)
        
        return self        
    
    def split(self, train_size=None, val_size=None, test_size=None, **options):
        from sklearn.model_selection import train_test_split
        
        n_samples = len(self.df.index)
        indices = np.arange(n_samples, dtype=int)
        
        if val_size is None or val_size == 0:
            indices_train, indices_test = train_test_split(indices, train_size=train_size, test_size=test_size, **options)
            indices_val = []
        else:
            if 0 < val_size < 1:
                val_size = int(val_size * n_samples)
            elif val_size < 0 or val_size > n_samples:
                raise ValueError(f'invalid val_size {val_size}. Expected a float in (0, 1), or a float in [0, n_samples={n_samples}].')
            
            if train_size is None:
                indices_train, indices_test = train_test_split(indices, test_size=test_size, **options)
                indices_train, indices_val = train_test_split(indices_train, test_size=val_size, **options)
            else:
                indices_train, indices_test = train_test_split(indices, train_size=train_size, **options)
                indices_val, indices_test = train_test_split(indices_test, train_size=val_size, **options)
                
        self.df[self.set_column] = pd.Series(pd.concat((
            pd.Series(data=self.train_key, index=indices_train),
            pd.Series(data=self.val_key, index=indices_val),
            pd.Series(data=self.test_key, index=indices_test)
        )), dtype='category')
        
    # def transform_samples(self, transformer, forward_self=False):
    #     if forward_self:
    #         return self.transform_samples(transformer(self), forward_self=False)
        
    #     self.df = self.apply(transformer())
    
class ImageDatasetTransformer:
    def __init__(self, transformers, persist_intermediate=False, persist_last=True, auto_purge=False):
        # transformers is in iterable yielding (path_transformer, value_transformer)
        
        self.transformers = transformers
        self._persist_intermediate = persist_intermediate
        self._persist_last = persist_last
        self._pipe = None
        self._assembled = False
        self._auto_purge = auto_purge
    
    def _assemble(self):
        import more_itertools as mit
        from functools import partial
        
        if self._persist_intermediate:
            transformers = mit.intersperse(Persistent.persist, self.transformers)        
        else:
            transformers = self.transformers

        transformers = mit.prepend(
            partial(load_persistent, auto_purge=self._auto_purge),
            transformers
        )
        
        if self._persist_last:
            transformers = bl.utils.append(transformers, Persistent.persist)
        
        self._assembled = True
        self._pipe = TransformationPipeline(*transformers)
    
    def transform_images(self, images, **kwargs):
        if not self._assembled:
            self._assemble()
        return self._pipe.transform(images, many=True, **kwargs)

# image_dataset_transformer = ImageDatasetTransformer(
#     (
#         PersistentTransformer(
#             'grayscale',
#             path_transformer=mover(python_project_home_path / 'testing_process_grayscale'),
#             value_transformer=skimage.color.rgb2gray,
#             lazy=True
#         ),
#         PersistentTransformer(
#             'crop',
#             path_transformer=mover(python_project_home_path / 'testing_process_grayscale_crop'),
#             value_transformer=functools.partial(bl.utils.image.crop, top=300, bottom=300, left=400, right=400),
#             lazy=True
#         ),
#         PersistentTransformer(
#             'flip',
#             path_transformer=mover(python_project_home_path / 'testing_process_grayscale_crop_flip'),
#             value_transformer=auto_gen(ArgGenerator(
#                 lambda *args, **kwargs: functools.partial(
#                     bl.utils.image.flip,
#                     *args,
#                     **kwargs    
#                 ),
#                 lambda *args, **kwargs: ([], {'horizontal': random_coin()}),
#                 auto_generate=True
#             )),
#             share_path=True,
#             lazy=True
#         ),
#     ),
#     persist_intermediate=True,
#     persist_last=True
# )
# image_dataset_transformer(
#     (python_project_home_path / 'testing_process').glob('*.jpg')
# )


def sync_to_img_ds(path_transformer, img_ds):
    def wrapped(old_path):
        new_path = path_transformer(old_path)
        img_ds.modify_path(old_path, new_path, many=False)
        return new_path
    return wrapped

# image_dataset_transformer = lambda img_ds: ImageDatasetTransformer(
#     (
#         PersistentTransformer(
#             'crop',
#             path_transformer=sync_to_img_ds(mover(python_project_home_path / 'crop', head_aggregator='_'), img_ds),
#             value_transformer=functools.partial(bl.utils.image.crop, top=300, bottom=300, left=400, right=400),
#             lazy=True
#         ),
#         PersistentTransformer(
#             'grayscale',
#             path_transformer=sync_to_img_ds(mover(python_project_home_path / 'grayscale', head_aggregator='_'), img_ds),
#             value_transformer=skimage.color.rgb2gray,
#             lazy=True
#         ),
#         PersistentTransformer(
#             'flip',
#             path_transformer=sync_to_img_ds(mover(python_project_home_path / 'flip', head_aggregator='_'), img_ds),
#             value_transformer=auto_gen(ArgGenerator(
#                 lambda *args, **kwargs: functools.partial(
#                     bl.utils.image.flip,
#                     *args,
#                     **kwargs
#                 ),
#                 lambda *args, **kwargs: ([], {'horizontal': random_coin()}),
#                 auto_generate=True
#             )),
#             share_path=True,
#             lazy=True
#         ),
#     ),
#     persist_intermediate=True,
#     persist_last=True
# )
        

# csv_path = 'my_csv.csv'
# df = pd.DataFrame(
#     {
#         'path': [
#             python_project_home_path / 'testing_process' / 'kiskadee.jpg',
#             python_project_home_path / 'testing_process' / 'kiskadee2.jpg'
#         ],
#         'set': pd.Series(['train', 'train'], dtype='category'),
#         'target': pd.Series(['clear', 'dark'], dtype='category'),
#     }
# )
# df.to_csv(csv_path, index=False)

# dataset = ImageDataset(csv_path, path_column='path', set_column='set', target_column='target')
# dataset.load()

# print(dataset.df)
# dataset.transform_images(image_dataset_transformer(dataset))
# print(dataset.df)
    
class Case:
    def __init__(
        self,
        path=None,
        root_path=None,
        name=None,
        videos_dir_name='videos',
        audios_dir_name='audios',
        frames_dir_name='frames',
        video_format='mp4',
        audio_format='m4a',
        frame_format='png',
        video_data_path=None
    ):
        if path is None:
            self.path = Path(root_path) / name
            self.name = name
        else:
            self.path = Path(path)
            self.name = self.path.name 
            
        self.videos_dir_name = videos_dir_name
        self.audios_dir_name = audios_dir_name
        self.frames_dir_name = frames_dir_name
            
        self.video_format = video_format
        self.audio_format = audio_format
        self.frame_format = frame_format
        
        if video_data_path is None:
            video_data_path = self.videos_path / 'data.json'
        self.video_data_path = video_data_path
    
    @property
    def videos_path(self):
        return self.path / self.videos_dir_name
            
    @property
    def audios_path(self):
        return self.path / self.audios_dir_name
    
    @property
    def frames_path(self):
        return self.path / self.frames_dir_name

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
        
        from more_itertools import flatten
        
        if glob_patterns is not None:
            return flatten(self.glob_videos(glob_pattern=glob_pattern) for glob_pattern in glob_patterns)
        
        return self.videos_path.glob(glob_pattern)
    
    def glob_audios(self, glob_pattern=None, glob_patterns=None):
        # TODO: use check_value_match
        
        from more_itertools import flatten
        
        if glob_patterns is not None:
            return flatten(self.glob_audios(glob_pattern=glob_pattern) for glob_pattern in glob_patterns)
        
        return self.audios_path.glob(glob_pattern)
    
    def glob_frames(self, glob_pattern=None, glob_patterns=None):
        # TODO: use check_value_match
        
        from more_itertools import flatten
        
        if glob_patterns is not None:
            return flatten(self.glob_frames(glob_pattern=glob_pattern) for glob_pattern in glob_patterns)
        
        return self.frames_path.glob(glob_pattern)

    @property
    def subcase_list(self):
        return [video_path.stem for video_path in self.video_path_list]
        
    @property
    def subcase_frames_dict(self):
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
                if verbose:
                    print(f'Audio does not exist. Extracting...', end=' ')
                bl.utils.video.extract_audio(video_path, audio_path)
                if verbose:
                    print(f'Done.')
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
                if verbose:
                    print(f'Frames do not exist or are incomplete. Extracting...', end=' ')
                    
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
                if verbose:
                    print(f'Done.')
            elif verbose:
                print(f'Frames already extracted. Skipping proccess.')
                
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
        from pandas import read_csv
        
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
        
        
# case = Case(root_path=python_project_home_path / 'cases', name='case 1')
# case.frames_path.mkdir(exist_ok=True, parents=True)
# case.audios_path.mkdir(exist_ok=True, parents=True)
# case.extract_audios()
# case.extract_frames()

# df = case.as_dataframe(predefined_column_types={'subcase': 'category', 'nominal_heat_flux': 'category'})
# bl.utils.print_header('case.df')
# print(df)

# ds = case.as_dataset(
#     predefined_column_types={'subcase': 'category', 'nominal_heat_flux': 'category'},
#     target_column='nominal_heat_flux'
# )

def plot_experiment(
    x_axis,
    experiment_dir=None,
    experiment_data_path=None,
    experiment_data_filename='data.csv',
    out_plot_dir=None,
    exclude_columns=None,
    label_filters=None,
    filter_as_subcase=False,
):
    from itertools import zip_longest
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    
    if experiment_data_path is None:
        experiment_data_path = experiment_dir / experiment_data_filename
    else:
        experiment_dir = experiment_data_path.parent
    
    if out_plot_dir is None:
        out_plot_dir = experiment_dir / 'plots'
    out_plot_dir.mkdir(exist_ok=True, parents=True)
        
    if exclude_columns is None:
        exclude_columns = set()
        
    if label_filters is None:
        label_filters = [
            ('Original data', bl.utils.packed_functional.identity)
        ]
        
    header = pd.read_csv(experiment_data_path, nrows=1).columns
    for column in set(header) - (set(exclude_columns) | {x_axis}):
        data = pd.read_csv(experiment_data_path, usecols=[x_axis, column])
        x, y = data[x_axis], data[column]
        
        if filter_as_subcase:
            for t in label_filters:
                if len(t) == 2:
                    label, f = t
                elif len(t) == 3:
                    label, f, column_list = t
                    if column not in column_list:
                        continue
                    
                plt.figure()
                plt.plot(x, f(y.to_numpy()))

                subcase_out_dir = out_plot_dir / column.replace('/', ' per ')
                subcase_out_dir.mkdir(exist_ok=True, parents=True)
                plt.savefig((subcase_out_dir / label).with_suffix('.png'))
                plt.close()
        else:
            plt.figure()
            for t in label_filters:
                if len(t) == 2:
                    label, f = t
                elif len(t) == 3:
                    label, f, column_list = t
                    if column not in column_list:
                        continue
                plt.plot(x, f(y.to_numpy()), label=label)
            
            plt.legend()
            plt.savefig((out_plot_dir / column.replace('/', ' per ')).with_suffix('.png'))
            plt.close()

from pprint import pprint

# label_filters = (
#     ('Original data', bl.utils.packed_functional.identity),
#     ('Moving average (n=11)', functools.partial(bl.utils.filters.smooth_mean, size=11)),
#     ('Clipped (200 W per cm2)', lambda x: np.minimum(x, 200), {'Flux [W/cm^2]'}),
# )
# for experiment_name in (
#     'Experiment Output 2020-03-17 11-51 (0)',
#     'Experiment Output 2020-03-17 16-49 (0)',
# ):
#     for filter_as_subcase in (True, False):
#         pprint({'Experiment name': experiment_name, 'filter_as_subcase': filter_as_subcase})
        
#         plot_experiment(
#             x_axis = 'Elapsed time',
#             experiment_data_path=python_project_home_path / 'experiments' / experiment_name / 'data.csv',
#             exclude_columns={'Time instant', 'Flux [W/m^2]'},
#             label_filters=label_filters,
#             filter_as_subcase=filter_as_subcase
#         )
    