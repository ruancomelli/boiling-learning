from pathlib import Path
import json

import utils

# TODO: check https://www.mlflow.org/docs/latest/tracking.html out

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
        save_method=None,
        load_method=None,
        verbose=0,
        printer=None,
    ):
        def default_save_method(model, path):
            import pickle
            with path.open('wb') as file:
                pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

        def default_load_method(path):
            import pickle
            with path.open('rb') as file:
                return pickle.load(file)

        if models_path is None:
            models_path = Path('.') / 'models'
        self.models_path = Path(models_path).resolve()

        if table_path is None:
            table_path = (models_path / 'lookup_table.json').resolve()
        self.table_path = table_path

        self.encoding = encoding

        if load_table:
            self.load_lookup_table()
        
        self.file_name_fmt = file_name_fmt
        self.creator_method = creator_method
        self.creator_name = creator_name

        if save_method is None:
            self.save_method = default_save_method
        else:
            self.save_method = save_method

        if load_method is None:
            self.load_method = default_load_method
        else:
            self.load_method = load_method
            
        self.verbose = verbose
        
        if printer is None:
            self.printer = print
        else:
            self.printer = printer

    def initialize_lookup_table(self, table_path=None):
        import json
        
        self.lookup_table = dict(files={})
        
        return self.save_lookup_table(table_path=table_path)

    def save_lookup_table(self, table_path=None):
        import json
        
        if table_path is not None:
            self.table_path = table_path
        
        self.table_path.parent.mkdir(parents=True, exist_ok=True)
        with self.table_path.open('w', encoding=self.encoding) as lookup_table_file:
            json.dump(self.lookup_table, lookup_table_file, indent=4, ensure_ascii=False)
            
        return self

    def load_lookup_table(self):
        import json
        
        if not self.table_path.exists():
            self.initialize_lookup_table()
        with self.table_path.open('r', encoding=self.encoding) as lookup_table_file:
            try:
                self.lookup_table = json.load(lookup_table_file)
            except json.JSONDecodeError:
                self.initialize_lookup_table()
                return self.load_lookup_table()
        
        return self

    def model_path(self, params, include=True):
        import parse
        
        def try_to_cast(obj, type):
            try:
                return (True, type(obj))
            except ValueError:
                return (False, obj)

        file_dict = self.lookup_table['files']
        for file_name, content in file_dict.items():
            if utils.json_equivalent(content, params):
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
            file_dict[key] = params

        self.save_lookup_table()
        
        return file_path

    def delete_model(self, params):
        for file_name, content in self.lookup_table['files'].items():
            if utils.json_equivalent(content, params):
                break

        self.lookup_table['files'].pop(file_name)
        self.model_path(params).unlink()

    def save_model(self, model, path=None, params=None):
        if path is None:
            path = self.model_path(params)

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
        hidden_params=None,
        save: bool = False,
        load: bool = False):

        from management import Mirror
        import pickle
        
        if creator_method is None:
            creator_method = self.creator_method

        if description is None:
            description = {}
        if hidden_params is None:
            hidden_params = {}
        if params is None:
            params = {}
        elif isinstance(params, Mirror):
            description = utils.merge_dicts(description, params['desc'])
            params = params['value']
        
        if self.verbose >= 2:
            self.printer('Description:', description)
            self.printer('Params:', params)
            self.printer('Hidden params:', hidden_params)
            
        if creator_name is None:
            if hasattr(creator_method, 'creator_name') and creator_method.creator_name is not None:
                creator_name = creator_method.creator_name
            elif self.creator_name is not None:
                creator_name = self.creator_name
            else:
                creator_name = repr(creator_method)
                
        try:
            description, final_params = creator_method.split(description, params, hidden_params)
        except AttributeError:
            description = utils.merge_dicts(description, params)
            final_params = utils.merge_dicts(params, hidden_params)
        
        path = self.model_path({
            'creator': creator_name,
            'parameters': description
        })

        if self.verbose >= 1:
            self.printer(f'Model path = {path}')

        if load:
            try:
                if self.verbose >= 1:
                    self.printer('Trying to load')
                return self.load_model(path)
            except (FileNotFoundError, pickle.UnpicklingError, OSError):
                if self.verbose >= 1:                
                    self.printer('Load failed')

        model = creator_method(final_params)

        if save:
            self.save_model(model, path)

        return model


def save_serialized(save_map):
    def save(return_dict, path):
        for key, obj in return_dict.items():
            path_ = path.with_suffix(path.suffix + '.' + key)
            save_map[key](obj, path_)
    return save

def save_keras_model(keras_model, path):
    keras_model.save(str(path.absolute().resolve()))

def save_pkl(obj, path):
    import pickle

    with path.absolute().resolve().open('wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_serialized(load_map):
    def load(path):
        loaded = {}
        for key, loader in load_map.items():
            path_ = path.with_suffix(path.suffix + '.' + key)
            loaded[key] = loader(path_)
            
        return loaded
    return load

def load_keras_model(path):
    from tensorflow.keras.models import load_model

    return load_model(
        str(path.absolute().resolve())
    )

def load_pkl(path):
    import pickle

    with path.absolute().resolve().open('rb') as file:
        return pickle.load(file)