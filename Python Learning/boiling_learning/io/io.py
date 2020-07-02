from pathlib import Path
import json
import pickle

import h5py
from tensorflow.keras.models import load_model

from boiling_learning.utils import (
    nullcontext,
    ensure_resolved,
    ensure_parent
)

def save_serialized(save_map):
    def save(return_dict, path):
        path = ensure_parent(path)
        for key, obj in return_dict.items():
            path_ = path / key
            save_map[key](obj, path_)
    return save

def save_keras_model(keras_model, path, **kwargs):
    path = ensure_parent(path)
    
    keras_model.save(path, **kwargs)

def save_pkl(obj, path):
    path = ensure_parent(path)

    with path.open('wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
        
def save_json(obj, path):
    path = ensure_parent(path)

    with path.open('w', encoding='utf-8') as file:
        json.dump(obj, file, indent=4, ensure_ascii=False)

def load_serialized(load_map):
    def load(path):
        loaded = {}
        path = Path(path)
        for key, loader in load_map.items():
            path_ = path / key
            loaded[key] = loader(path_)
            
        return loaded
    return load

def load_keras_model(path, strategy=None, **kwargs):
    if strategy is None:
        scope = nullcontext()
    else:
        scope = strategy.scope()
        
    with scope:
        return load_model(path, **kwargs)

def load_pkl(path):
    with ensure_resolved(path).open('rb') as file:
        return pickle.load(file)

def load_json(path):
    with ensure_resolved(path).open('r', encoding='utf-8') as file:
        return json.load(file)

def saver_hdf5(key=''):
    def save_hdf5(obj, path):
        path = ensure_parent(path)
        with h5py.File(str(path), 'w') as hf:
            hf.create_dataset(key, data=obj)
    return save_hdf5

def loader_hdf5(key=''):
    def load_hdf5(path):
        path = ensure_resolved(path)
        with h5py.File(str(path), 'r') as hf:
            return hf.get(key)
    return load_hdf5
            
            