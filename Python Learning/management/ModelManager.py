from pathlib import Path

import utils

# TODO: check out <https://www.mlflow.org/docs/latest/tracking.html>

def save_serialized(save_map):
    def save(return_dict, path):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        for key, obj in return_dict.items():
            path_ = path / key
            save_map[key](obj, path_)
    return save

def save_keras_model(keras_model, path, **kwargs):
    from pathlib import Path
    
    path = Path(path).absolute().resolve()
    path.mkdir(exist_ok=True, parents=True)
    
    keras_model.save(path, **kwargs)

def save_pkl(obj, path):
    import pickle

    path = Path(path).absolute().resolve()

    with path.open('wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

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
    from tensorflow.keras.models import load_model

    return load_model(
        path,
        **kwargs
    )

def load_pkl(path):
    import pickle

    with path.absolute().resolve().open('rb') as file:
        return pickle.load(file)

class ModelManager:
    def __init__(
        self,
        models_path=None,
        table_path=None,
        encoding='utf-8',
        load_table=True,
        file_name_fmt='{index}.data',
        creator_method=None,
        creator_name=None,
        save_method=save_pkl,
        load_method=load_pkl,
        verbose=0,
        printer=print,
    ):
        if models_path is None:
            models_path = Path() / 'models'
        self.models_path = Path(models_path).resolve()

        if table_path is None:
            table_path = (self.models_path / 'lookup_table.json').resolve()
        self.table_path = Path(table_path).resolve()
        self.encoding = encoding

        if load_table:
            self.load_lookup_table()
        
        self.file_name_fmt = file_name_fmt
        self.creator_method = creator_method
        self.creator_name = creator_name
        self.save_method = save_method
        self.load_method = load_method
        self.verbose = verbose
        self.printer = printer

    def _initialize_lookup_table(self):
        self.lookup_table = dict(files={})
        
        return self.save_lookup_table()

    def save_lookup_table(self):
        import json
        
        self.table_path.parent.mkdir(parents=True, exist_ok=True)
        with self.table_path.open('w', encoding=self.encoding) as lookup_table_file:
            json.dump(self.lookup_table, lookup_table_file, indent=4, ensure_ascii=False)
            
        return self

    def load_lookup_table(self):
        import json
        
        if not self.table_path.exists():
            self._initialize_lookup_table()
        with self.table_path.open('r', encoding=self.encoding) as lookup_table_file:
            try:
                self.lookup_table = json.load(lookup_table_file)
            except json.JSONDecodeError:
                self._initialize_lookup_table()
                return self.load_lookup_table()
        
        return self
    
    def _merge_description_and_creator_name(self, description, creator_name):
        return {
            'creator': creator_name,
            'parameters': description
        }
        
    def model_path(self, description, creator_name=None, include: bool = True):        
        import parse
        
        if creator_name is not None:
            return self.model_path(
                description=self._merge_description_and_creator_name(
                    description=description,
                    creator_name=creator_name
                ),
                include=include
            )
        
        def try_to_cast(obj, type):
            try:
                return (True, type(obj))
            except ValueError:
                return (False, obj)

        file_dict = self.lookup_table['files']
        for file_name, content in file_dict.items():
            if utils.json_equivalent(content, description):
                break
        else:
            pattern = parse.compile(self.file_name_fmt)
            
            parsed_items = (pattern.parse(key) for key in file_dict)
            indices = (parsed['index'] for parsed in parsed_items if 'index' in parsed)
            int_key_list = [
                cast[1]
                for cast in map(
                    lambda x: try_to_cast(x, int),
                    indices
                )
                if cast[0]
            ]
            
            missing_elems = utils.missing_elements(int_key_list)
            if missing_elems:
                index = missing_elems[0]
            else:
                index = str(max(int_key_list, default=-1) + 1)
            file_name = self.file_name_fmt.format(index=index)

        file_path = (self.models_path / file_name).resolve().absolute()
        
        key = utils.relative_path(self.table_path.parent, file_path)

        if include:
            file_dict[key] = description

        self.save_lookup_table()
        
        return file_path

    def delete_model(self, description):
        for file_name, content in self.lookup_table['files'].items():
            if utils.json_equivalent(content, description):
                break

        self.lookup_table['files'].pop(file_name)
        self.model_path(description).unlink()

    def save_model(self, model, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_method(model, path)
        return self

    def load_model(self, path):
        return self.load_method(path)
    
    def provide_model(
        self,
        creator_method=None,
        creator_name=None,
        description=None,
        params=None,
        save: bool = False,
        load: bool = False,
        raise_if_load_fails: bool = False
    ):
        from pickle import UnpicklingError
        
        if creator_method is None:
            creator_method = self.creator_method

        if description is None:
            description = {}
        if params is None:
            params = {}
        
        if self.verbose >= 2:
            self.printer('Description:', description)
            self.printer('Params:', params)
            
        if creator_name is None:
            if hasattr(creator_method, 'creator_name'):
                creator_name = creator_method.creator_name
            elif self.creator_name is not None:
                creator_name = self.creator_name
            else:
                creator_name = str(creator_method)
        
        path = self.model_path(description=description, creator_name=creator_name)

        if self.verbose >= 1:
            self.printer(f'Model path = {path}')

        if load:
            try:
                if self.verbose >= 1:
                    self.printer('Trying to load')
                return self.load_model(path)
            except (FileNotFoundError, UnpicklingError, OSError):
                if self.verbose >= 1:                
                    self.printer('Load failed')
                if raise_if_load_fails:
                    raise

        model = creator_method(params)

        if save:
            self.save_model(model, path)

        return model