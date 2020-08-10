from pathlib import Path
from json import JSONDecodeError
import operator
from typing import (
    overload,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union
)
import copy
import collections

import parse
import more_itertools as mit

import boiling_learning as bl
from boiling_learning.management.Parameters import Parameters

# TODO: check out <https://www.mlflow.org/docs/latest/tracking.html>

_sentinel = object()

PathType = bl.utils.PathType
VerboseType = bl.utils.VerboseType


class Manager(
        bl.utils.SimpleRepr,
        bl.utils.SimpleStr,
        collections.abc.Mapping
):
    def __init__(
            self,
            path: PathType,
            id_fmt: str = '{index}.data',
            index_key: str = 'index',
            save_method: Callable[[Any, PathType], Any] = bl.io.save_pkl,
            load_method: Callable[[PathType], Any] = bl.io.load_pkl,
            verbose: VerboseType = False,
            load_table: bool = True,
    ):
        '''
        The Manager's directory is structure like this:
        path/
            lookup_table.json
            shared/
            entries/
                1.data/
                    model/
                    workspace/
                2.data
                    model/
                    workspace/
                ...

        The `lookup_table` is structured like this:

        ```
        {
            'entries': { # entry_key
                *model_id*: {
                   'model': { # data_key: contents
                       'creator': *creator_name*, # creator_key
                       'description': *description_dict* # description_key
                   }
                   'metadata': *metadata_dict*  # metadata_key: metadata
                }
                ...
            }
        }
        ```

        '''
        self.entries_key = 'entries'
        self.contents_key = 'model'
        self.metadata_key = 'metadata'
        self.creator_key = 'creator'
        self.description_key = 'parameters'
        self._workspace_key = 'workspace'

        self._path = bl.utils.ensure_dir(path)
        self._table_path = self.path / 'lookup_table.json'
        self._entries_dir_path = bl.utils.ensure_dir(
            self.path / self.entries_key)
        self._shared_dir_path = bl.utils.ensure_dir(self.path / 'shared')

        if load_table:
            self.load_lookup_table()

        self.id_fmt = id_fmt
        self.index_key = index_key
        self.path_key = 'path'

        self._save_method = save_method
        self._load_method = load_method
        self.verbose = verbose

    def __getitem__(self, model_id: str):
        self.load_lookup_table()
        return self._lookup_table.setdefault(self.entries_key, {})[model_id]

    def __setitem__(self, model_id: str, entry: Mapping[str, Any]) -> None:
        self._lookup_table.setdefault(self.entries_key, {})[model_id] = entry
        self.save_lookup_table()

    def __delitem__(self, model_id: str) -> None:
        del self._lookup_table.setdefault(self.entries_key, {})[model_id]
        self.save_lookup_table()

    def __iter__(self) -> Iterator:
        return self._lookup_table.get(self.entries_key, {}).__iter__()

    def __len__(self) -> int:
        return self._lookup_table.get(self.entries_key, {}).__len__()

    @property
    def entries(self) -> dict:
        return self._lookup_table[self.entries_key]

    @property
    def path(self) -> Path:
        return self._path

    @property
    def table_path(self) -> Path:
        return self._table_path

    @property
    def entries_dir(self) -> Path:
        return self._entries_dir_path

    @property
    def shared_dir(self) -> Path:
        return self._shared_dir_path

    def entry_dir(self, model_id: str) -> Path:
        return bl.utils.ensure_dir(self.entries_dir / model_id)

    def model_path(self, model_id: str) -> Path:
        return bl.utils.ensure_parent(self.entry_dir(model_id) / self.contents_key)

    def model_workspace(self, model_id: str) -> Path:
        return bl.utils.ensure_dir(self.entry_dir(model_id) / self._workspace_key)

    def _initialize_lookup_table(self) -> None:
        self._lookup_table = {
            self.entries_key: {}
        }
        self.save_lookup_table()

    def save_lookup_table(self) -> None:
        bl.io.save_json(self._lookup_table, self._table_path)
        return self

    def load_lookup_table(
        self,
        raise_if_fails: bool = False
    ) -> None:
        if not self._table_path.is_file():
            self._initialize_lookup_table()
        try:
            self._lookup_table = bl.io.load_json(self._table_path)
        except (FileNotFoundError, JSONDecodeError, OSError):
            if raise_if_fails:
                raise
            else:
                self._initialize_lookup_table()
                self.load_lookup_table()

    def save_model(self, model, path: PathType) -> None:
        path = bl.utils.ensure_parent(path)
        self._save_method(model, path)

    def load_model(self, path: PathType):
        path = bl.utils.ensure_resolved(path)
        return self._load_method(path)

    @overload
    def contents(self, model_id: None) -> dict: ...
    @overload
    def contents(self, model_id: str): ...

    def contents(self, model_id=None):
        if model_id is None:
            return {
                model_id: self.contents(model_id)
                for model_id in self.entries
            }
        else:
            return self.entries.get(model_id, {}).get(self.contents_key)

    @overload
    def metadata(self, model_id: None) -> dict: ...
    @overload
    def metadata(self, model_id: str): ...

    def metadata(self, model_id=None):
        if model_id is None:
            return {
                model_id: self.metadata(model_id)
                for model_id in self.entries
            }
        else:
            return self.entries.get(model_id, {}).get(self.metadata_key, {})

    @overload
    def model_creator(self, model_id: None) -> dict: ...
    @overload
    def model_creator(self, model_id: str): ...

    def model_creator(self, model_id=None):
        if model_id is None:
            return {
                model_id: self.model_creator(model_id)
                for model_id in self.entries
            }
        else:
            return self.contents(model_id).get(self.creator_key)

    @overload
    def description(self, model_id: None) -> dict: ...
    @overload
    def description(self, model_id: str): ...

    def description(self, model_id=None):
        if model_id is None:
            return {
                model_id: self.description(model_id)
                for model_id in self.entries
            }
        else:
            return self.contents(model_id).get(self.description_key, {})

    def _parse_index(self, model_id) -> int:
        return int(parse.parse(self.id_fmt, model_id)[self.index_key])

    def _format_index(self, index) -> str:
        return self.id_fmt.format(**{self.index_key: index})

    def new_model_id(self) -> str:
        '''Return a model id that does not exist yet
        '''
        indices = sorted(
            mit.map_except(
                self._parse_index,
                self.entries.keys(),
                ValueError, TypeError, KeyError, AttributeError
            )
        )
        if indices:
            missing_elems = bl.utils.missing_ints(indices)
            index = mit.first(
                missing_elems,
                indices[-1] + 1
            )
        else:
            index = 0

        return self._format_index(index)

    def _resolve_creator_name(
            self,
            contents: Optional[Mapping] = None,
            creator: Optional[Callable] = None,
            creator_name: Optional[str] = None
    ) -> str:
        if creator_name is not None:
            return creator_name
        elif contents is not None and self.creator_key in contents:
            return contents[self.creator_key]
        # support creator type
        elif hasattr(creator, 'creator') and hasattr(creator.creator, 'creator_name'):
            return creator.creator.creator_name
        elif hasattr(creator, 'creator_name'):  # support model creator
            return creator.creator_name
        else:
            return str(creator)

    def _resolve_contents(
            self,
            contents: Optional[Mapping] = None,
            description: Optional[Mapping] = None,
            creator: Optional[Callable] = None,
            creator_name: Optional[str] = None
    ) -> Mapping:
        if contents is not None:
            return contents
        else:
            creator_name = self._resolve_creator_name(
                contents=contents, creator=creator, creator_name=creator_name)

            return {
                self.creator_key: creator_name,
                self.description_key: description
            }

    def _resolve_metadata(
            self,
            metadata: Optional[Mapping] = None,
            path: Optional[PathType] = None
    ) -> Mapping:
        # TODO: check for erros e.g. when metadata and path are both None
        if metadata is None:
            return {
                self.path_key: path
            }
        else:
            return metadata

    def _resolve_entry(
            self,
            contents: Optional[Mapping] = None,
            description: Optional[Mapping] = None,
            creator: Optional[Callable] = None,
            creator_name: Optional[str] = None,
            metadata: Optional[Mapping] = None,
            path: Optional[PathType] = None
    ) -> Mapping:
        contents = self._resolve_contents(
            contents=contents, description=description, creator=creator, creator_name=creator_name)
        metadata = self._resolve_metadata(metadata=metadata, path=path)

        if path is None:
            raise TypeError(f'invalid model path: {path}')

        return {
            self.contents_key: contents,
            self.metadata_key: {
                self.path_key: path
            }
        }

    def model_id(
            self,
            contents: Optional[Mapping] = None,
            description: Optional[Mapping] = None,
            creator: Optional[Callable] = None,
            creator_name: Optional[str] = None,
            missing_ok: bool = True
    ) -> str:
        contents = self._resolve_contents(
            contents=contents, description=description, creator=creator, creator_name=creator_name)

        candidates = bl.utils.extract_keys(
            self.contents(),
            value=contents,
            cmp=bl.utils.json_equivalent
        )
        model_id = mit.only(candidates, default=_sentinel)

        if model_id is not _sentinel:
            return model_id
        elif missing_ok:
            return self.new_model_id()
        else:
            raise ValueError(
                f'could not find model with the following contents: {contents}')

    def has_model(
            self,
            contents: Optional[Mapping] = None,
            description: Optional[Mapping] = None,
            creator: Optional[Callable] = None,
            creator_name: Optional[str] = None,
    ) -> bool:
        try:
            return self.model_id(
                contents=contents,
                description=description,
                creator=creator,
                creator_name=creator_name,
                missing_ok=False
            ) in self.entries
        except ValueError:
            return False

    def _repeated_models(self) -> Tuple[Dict[str, List[str]], List[str]]:
        all_contents = self.contents().items()

        repetition_dict = {
            model_id: sorted(
                {
                    other_id
                    for other_id, other_contents in all_contents
                    if other_id != model_id and bl.utils.json_equivalent(contents, other_contents)
                },
                key=self._parse_index
            )
            for model_id, contents in all_contents
        }

        repeated = sorted(
            {
                x
                for lst in repetition_dict.values()
                for x in lst[1:]
            },
            key=self._parse_index
        )

        return repetition_dict, repeated

    def _retrieve_model(
            self,
            path: PathType,
            raise_if_load_fails: bool
    ) -> Tuple[bool, Any]:
        try:
            return True, self.load_model(path)
        except tuple(getattr(self._load_method, 'expected_exceptions', (FileNotFoundError, OSError))):
            if self.verbose >= 1:
                print('Load failed')
            if raise_if_load_fails:
                raise
            return False, None

    def update_entries(
            self,
            updater: Callable[[str, Parameters], Any],
            model_ids: Optional[Iterable[str]] = None,
            save: bool = True
    ) -> None:
        if model_ids is None:
            model_ids = self.entries

        self.load_lookup_table()

        entries = self.entries
        for model_id in model_ids:
            old_entry = copy.deepcopy(entries[model_id])
            new_entry = updater(model_id, old_entry)

            self[model_id] = new_entry

        if save:
            self.save_lookup_table()

    def retrieve_model(
            self,
            contents: Optional[Mapping] = None,
            description: Optional[Mapping] = None,
            creator: Optional[Callable] = None,
            creator_name: Optional[str] = None,
            raise_if_load_fails: bool = False,
    ) -> Tuple[bool, Any]:
        if not self.has_model(contents=contents, description=description, creator=creator, creator_name=creator_name):
            return False, None

        model_id = self.provide_entry(
            contents=contents,
            description=description,
            creator=creator,
            creator_name=creator_name,
            include=False,
            missing_ok=False
        )

        path = self.model_path(model_id)

        return self._retrieve_model(
            path=path,
            raise_if_load_fails=raise_if_load_fails
        )

    def retrieve_models(
            self,
            entry_pred: Optional[Callable[[Mapping], bool]] = None
    ) -> Iterable[Tuple[str, Any]]:
        if entry_pred is None:
            models = self.entries.keys()
        else:
            models = (
                model_id
                for model_id, entry in self.entries.items()
                if entry_pred(entry)
            )

        models = (
            (model_id, self.model_path(model_id))
            for model_id in models
        )

        models = (
            (model_id, self._retrieve_model(path, raise_if_load_fails=False))
            for model_id, path in models
        )

        models = (
            (model_id, model)
            for model_id, (success, model) in models
            if success
        )

        return models

    def provide_entry(
            self,
            contents: Optional[Mapping] = None,
            description: Optional[Mapping] = None,
            creator: Optional[Callable] = None,
            creator_name: Optional[str] = None,
            include: bool = False,
            missing_ok: bool = False
    ) -> str:
        contents = self._resolve_contents(
            contents=contents, description=description, creator=creator, creator_name=creator_name)

        model_id = self.model_id(
            contents=contents,
            missing_ok=missing_ok
        )
        path = self.model_path(model_id)
        model_rel_path = bl.utils.relative_path(self.entries_dir, path)

        entry = self._resolve_entry(
            contents=contents,
            path=model_rel_path
        )

        if include:
            self[model_id] = entry

        return model_id

    def provide_model(
            self,
            contents: Optional[Mapping] = None,
            description: Optional[Mapping] = None,
            creator: Optional[Callable] = None,
            creator_name: Optional[str] = None,
            params=None,
            save: bool = False,
            load: bool = False,
            raise_if_load_fails: bool = False,
            unpack_parameters: bool = False,
    ):
        if params is None:
            params = {}

        if self.verbose >= 2:
            bl.utils.print_header('Description', level=1)
            print(description)
            bl.utils.print_header('Params', level=1)
            print(params)

        model_id = self.provide_entry(
            contents=contents,
            description=description,
            creator=creator,
            creator_name=creator_name,
            include=True,
            missing_ok=True
        )

        path = self.model_path(model_id)

        if load:
            success, model = self._retrieve_model(
                path=path, raise_if_load_fails=raise_if_load_fails)
            if success:
                return model

        if unpack_parameters:
            model = creator(**params)
        else:
            model = creator(params)

        if save:
            self.save_model(model, path)

        return model
