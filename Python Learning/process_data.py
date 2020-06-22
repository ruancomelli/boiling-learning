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

_sentinel = object()

class ArgGenerator:
    def __init__(self, f, generator, keyer=None, auto_generate=False):
        self._generator = generator
        self._store = dict()
        self._fun = bl.utils.functional.packed(f)
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
    
    def __call__(self, key=_sentinel):
        if key is _sentinel:
            return self._fun(bl.utils.functional.pack())
            
        if self._auto_generate:
            self.generate(key)
        return self._fun(self[key])

@bl.utils.constant_factory
def random_coin():
    from random import choice
    
    return choice([False, True])

def auto_gen(arg_generator, key_index=0):
    def wrapped(*args, **kwargs):
        args = list(args)
        key = args.pop(key_index)
        return arg_generator(key)(*args, **kwargs)
    return wrapped

def sync_to_img_ds(path_transformer, img_ds):
    def wrapped(old_path):
        new_path = path_transformer(old_path)
        img_ds.modify_path(old_path, new_path, many=False)
        return new_path
    return wrapped

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
            ('Original data', bl.utils.functional.identity)
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
