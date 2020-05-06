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

python_project_home_path = Path().absolute().resolve()
project_home_path = python_project_home_path.parent.resolve()

_marker = object()

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
#             value_transformer=functools.partial(bl.preprocessing.image.crop, top=300, bottom=300, left=400, right=400),
#             lazy=True
#         ),
#         PersistentTransformer(
#             'flip',
#             path_transformer=mover(python_project_home_path / 'testing_process_grayscale_crop_flip'),
#             value_transformer=auto_gen(ArgGenerator(
#                 lambda *args, **kwargs: functools.partial(
#                     bl.preprocessing.image.flip,
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
    