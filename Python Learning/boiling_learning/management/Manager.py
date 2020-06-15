from pathlib import Path
from json import JSONDecodeError
import operator

import parse
import more_itertools as mit

import boiling_learning as bl

# TODO: check out <https://www.mlflow.org/docs/latest/tracking.html>

_sentinel = object()

class Manager(bl.utils.SimpleRepr, bl.utils.SimpleStr):
    def __init__(
        self,
        root_path,
        data_path=None,
        table_path=None,
        load_table=True,
        id_fmt='{index}.data',
        index_key='index',
        data_key='models',
        path_key='path',
        creator_key='creator',
        description_key='parameters',
        save_method=bl.io.save_pkl,
        load_method=bl.io.load_pkl,
        verbose=0,
        printer=print,
    ):
        self.index_key = index_key
        self.data_key = data_key
        self.path_key = path_key
        self.creator_key = creator_key
        self.description_key = description_key
        
        self.root_path = Path(root_path)
        if data_path is None:
            data_path = self.root_path / self.data_key
        self.data_path = Path(data_path).resolve()

        if table_path is None:
            table_path = (self.data_path / 'lookup_table.json').resolve()
        self.table_path = Path(table_path).resolve()

        if load_table:
            self.load_lookup_table()
        
        self.id_fmt = id_fmt
        self.save_method = save_method
        self.load_method = load_method
        self.verbose = verbose
        self.printer = printer

    def _initialize_lookup_table(self):
        self.lookup_table = {
            self.data_key: {}
        }
        
        return self.save_lookup_table()

    def save_lookup_table(self):
        bl.io.save_json(self.lookup_table, self.table_path)
            
        return self

    def load_lookup_table(self, raise_if_fails: bool = False):
        if not self.table_path.exists():
            self._initialize_lookup_table()
        try:
            self.lookup_table = bl.io.load_json(self.table_path)
        except (FileNotFoundError, JSONDecodeError, OSError):
            if raise_if_fails:
                raise
            else:
                self._initialize_lookup_table()
                return self.load_lookup_table()
        
        return self

    def save_model(self, model, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_method(model, path)
        return self

    def load_model(self, path):
        return self.load_method(path)
        
    @property
    def entries(self):
        return self.lookup_table[self.data_key]
        
    @property
    def model_ids(self):
        return self.entries.keys()
    
    @property
    def contents(self):
        return {
            model_id: self._get_content(model_id)
            for model_id in self.model_ids
        }
        
    def new_model_id(self):
        '''Return a model id that does not exist yet
        '''
        pattern = parse.compile(self.id_fmt)
        parser = pattern.parse
        parsed_items = map(parser, self.model_ids) # parse all ids
        matched_items = filter(bool, parsed_items) # keep only those that have matched
        indexed_items = filter(lambda item: self.index_key in item, matched_items) # keep only those that have matched the index key
        indices = map(operator.itemgetter(self.index_key), indexed_items) # get the indices
        int_indices_list = sorted(mit.map_except(int, indices, ValueError, TypeError)) # cast indices to int
        
        if int_indices_list:
            missing_elems = bl.utils.missing_ints(int_indices_list)
            index = mit.first(
                missing_elems,
                int_indices_list[-1] + 1
            )
        else:
            index = 0
        
        return self.id_fmt.format(**{self.index_key: index})
    
    def _get_entry(self, model_id):
        return self.entries[model_id]
    
    def _get_content(self, model_id):
        entry = self._get_entry(model_id)
        return {
            key: entry[key]
            for key in (self.creator_key, self.description_key)
        }
        
    def _get_description(self, model_id):
        entry = self._get_entry(model_id)
        return entry[self.description_key]
        
    def _get_creator(self, model_id):
        entries = self._get_entry(model_id)
        return entries[self.creator_key]
        
    def _get_path(self, model_id):
        entries = self._get_entry(model_id)
        return entries[self.path_key]
    
    def _resolve_creator_name(self, content=None, creator=None, creator_name=None):
        if creator_name is not None:
            return creator_name
        elif content is not None and self.creator_key in content:
            return content[self.creator_key]
        elif hasattr(creator, 'creator') and hasattr(creator.creator, 'creator_name'): # support creator type
            return creator.creator.creator_name
        elif hasattr(creator, 'creator_name'): # support model creator
            return creator.creator_name
        else:
            return str(creator)
        
    def _resolve_description(self, content=None, description=None):
        if description is not None:
            return description
        else:
            return content[self.description_key]
        
    def _resolve_content(self, content=None, description=None, creator=None, creator_name=None):
        if content is not None:
            return content
        else:    
            creator_name = self._resolve_creator_name(content=content, creator=creator, creator_name=creator_name)
            
            return {
                self.creator_key: creator_name,
                self.description_key: description
            }
        
    def _resolve_entry(
        self,
        content=None,
        description=None,
        creator=None,
        creator_name=None,
        path=None
    ):
        content = self._resolve_content(content=content, description=description, creator=creator, creator_name=creator_name)
        
        if path is None:
            raise TypeError(f'invalid model path: {path}')
        
        return bl.utils.merge_dicts(
            {self.path_key: path},
            content,
            latter_precedence=False
        )
        
    def has_model(
        self,
        content=None,
        description=None,
        creator=None,
        creator_name=None
    ):
        return self.model_id(
            content=content,
            description=description,
            creator=creator,
            creator_name=creator_name,
            missing_ok=True            
        ) in self.model_ids
        
    def model_id(
        self,
        content=None,
        description=None,
        creator=None,
        creator_name=None,
        missing_ok: bool = True
    ):
        content = self._resolve_content(content=content, description=description, creator=creator, creator_name=creator_name)
        
        candidates = bl.utils.extract_keys(
            self.contents,
            value=content,
            cmp=bl.utils.json_equivalent
        )
        model_id = mit.only(candidates, default=_sentinel)
        
        if model_id is not _sentinel:
            return model_id
        elif missing_ok:
            return self.new_model_id()
        else:
            raise ValueError(f'could not find model with the following content: {content}')
    
    def model_path(
        self,
        content=None,
        description=None,
        creator=None,
        creator_name=None,
        include: bool = False,
        missing_ok: bool = False,
        full: bool = True
    ):
        content = self._resolve_content(content=content, description=description, creator=creator, creator_name=creator_name)

        model_id = self.model_id(
            content=content,
            missing_ok=missing_ok
        )
        model_full_path = (self.data_path / model_id).resolve().absolute()
        model_rel_path = bl.utils.relative_path(self.table_path.parent, model_full_path)
        
        if include:
            entry = self._resolve_entry(
                content=content,
                path=model_rel_path
            )
            self.entries[model_id] = entry
            self.save_lookup_table()
        
        if full:
            path = model_full_path
        else:
            path = model_rel_path
        if self.verbose >= 1:
            self.printer(f'Model path = {path}')
        return path
    
    def _retrieve_model(
        self,
        path,
        raise_if_load_fails
    ):
        try:
            return True, self.load_model(path)
        except tuple(getattr(self.load_method, 'expected_exceptions', (FileNotFoundError, OSError))):
            if self.verbose >= 1:
                self.printer('Load failed')
            if raise_if_load_fails:
                raise
            return False, None
    
    def retrieve_model(
        self,
        content=None,
        description=None,
        creator=None,
        creator_name=None,
        raise_if_load_fails: bool = False,
    ):        
        if not self.has_model(content=content, description=description, creator=creator, creator_name=creator_name):
            return False, None
        
        path = self.model_path(
            content=content,
            description=description,
            creator=creator,
            creator_name=creator_name,
            include=False,
            missing_ok=False,
            full=True
        )
        
        return self._retrieve_model(
            path=path,
            raise_if_load_fails=raise_if_load_fails
        )
        
    def retrieve_models(
        self,
        entry_pred=None
    ):
        path_key = self.path_key
        entries = self.entries.items()
        
        entries = (
            (model_id, entry)
            for model_id, entry in entries
            if path_key in entry
        )
        
        if entry_pred is not None:
            entries = (
                (model_id, entry)
                for model_id, entry in entries
                if entry_pred(entry)
            )
        
        entries = (
            (model_id, self.table_path.parent / entry[path_key])
            for model_id, entry in entries
        )
        
        entries = (
            (model_id, self._retrieve_model(path, raise_if_load_fails=False))
            for model_id, path in entries
        )
        
        entries = (
            (model_id, model)
            for model_id, (success, model) in entries
            if success
        )
        
        return entries
        
    def provide_model(
        self,
        content=None,
        description=None,
        creator=None,
        creator_name=None,
        params=None,
        save: bool = False,
        load: bool = False,
        raise_if_load_fails: bool = False,
        content_key=_sentinel,
    ):
        if params is None:
            params = {}
        
        if self.verbose >= 2:
            bl.utils.print_header('Description', level=1)
            self.printer(description)
            bl.utils.print_header('Params', level=1)
            self.printer(params)
        
        path = self.model_path(
            content=content,
            description=description,
            creator=creator,
            creator_name=creator_name,
            include=True,
            missing_ok=True,
            full=True
        )

        if load:
            success, model = self._retrieve_model(path=path, raise_if_load_fails=raise_if_load_fails)
            if success:
                return model

        model = creator(params)
        
        if content_key is not _sentinel:
            model[content_key] = content

        if save:
            self.save_model(model, path)

        return model