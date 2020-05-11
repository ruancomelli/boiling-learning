def save_serialized(save_map):
    def save(return_dict, path):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        for key, obj in return_dict.items():
            path_ = path / key
            save_map[key](obj, path_)
    return save

def save_keras_model(keras_model, path, **kwargs):
    path = Path(path).absolute().resolve()
    path.mkdir(exist_ok=True, parents=True)
    
    keras_model.save(path, **kwargs)

def save_pkl(obj, path):
    path = Path(path).absolute().resolve()
    path.mkdir(exist_ok=True, parents=True)

    with path.open('wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
        
def save_json(obj, path):
    path = Path(path).absolute().resolve()
    path.mkdir(exist_ok=True, parents=True)

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

def load_keras_model(path, **kwargs):
    return load_model(
        path,
        **kwargs
    )

def load_pkl(path):
    with Path(path).absolute().resolve().open('rb') as file:
        return pickle.load(file)

def load_json(path):
    with Path(path).absolute().resolve().open('r', encoding='utf-8') as file:
        return json.load(file)