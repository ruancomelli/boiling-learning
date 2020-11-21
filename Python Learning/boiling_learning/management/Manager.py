import copy
from dataclasses import dataclass
from functools import partial
import json
import operator
import pprint
from pathlib import Path
from typing import (
    overload,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union
)

import funcy
import more_itertools as mit
import parse

import boiling_learning as bl
from boiling_learning.utils.functional import (
    Pack
)
from boiling_learning.utils.utils import (
    # JSONDataType, # TODO: maybe using JSONDataType would be good
    PathType,
    VerboseType
)
from boiling_learning.io.io import (
    LoaderFunction,
    SaverFunction
)
from boiling_learning.preprocessing.transformers import Creator, Transformer
from boiling_learning.management.Parameters import Parameters


# TODO: check out <https://www.mlflow.org/docs/latest/tracking.html>
# TODO: maybe include "status" in metadata
# TODO: maybe include a comment in metadata
# TODO: improve this... there is a lot of repetition and buggy cases.
# ? Perhaps a better would be to have only one way to pass a description: through elem_id. No *contents*, just the id...


_sentinel = object()
T = TypeVar('T')
S = TypeVar('S')


class Manager(
        bl.utils.SimpleRepr,
        bl.utils.SimpleStr,
        Mapping[str, dict],
        Generic[T]
):
    @dataclass(frozen=True)
    class Keys:
        entries: str = 'entries'
        elements: str = 'model'
        metadata: str = 'metadata'
        creator: str = 'creator'
        creator_params: str = 'creator_params'
        post_processor: str = 'post_processor'
        post_processor_params: str = 'post_processor_params'
        workspace: str = 'workspace'
        path: str = 'path'

    # @dataclass(frozen=True)
    # class Entry:
    #     # TODO: here
    #     creator: str
    #     creator_params: PackType
    #     post_processor: str
    #     post_processor_params: PackType

    def __init__(
            self,
            path: PathType,
            id_fmt: str = '{index}.data',
            index_key: str = 'index',
            save_method: Optional[SaverFunction[T]] = None,
            load_method: Optional[LoaderFunction[T]] = None,
            creator: Optional[Creator[T]] = None,
            post_processor: Optional[Transformer[T, S]] = None,
            verbose: VerboseType = False,
            load_table: bool = True,
            keys: Keys = Keys(),
            json_encoder: json.JSONEncoder = json.JSONEncoder,
            json_decoder: json.JSONDecoder = json.JSONDecoder
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
            'entries': { // entry_key
                *elem_id*: {
                   'model': { // keys.elements: contents
                       'creator': *creator_name*, // keys.creator
                       'creator_params': *creator_params* // keys.creator_params
                   }
                   'metadata': *metadata_dict*  // keys.metadata: metadata
                }
                ...
            }
        }
        ```
        '''
        self.keys: self.Keys = keys
        self._path: Path = bl.utils.ensure_dir(path)
        self._table_path: Path = self.path / 'lookup_table.json'
        self._entries_dir_path: Path = bl.utils.ensure_dir(
            self.path / self.keys.entries)
        self._shared_dir_path: Path = bl.utils.ensure_dir(self.path / 'shared')
        self._json_encoder = json_encoder
        self._json_decoder = json_decoder

        if load_table:
            self.load_lookup_table()

        self.id_fmt: str = id_fmt
        self.index_key: str = index_key

        self._parse_index: Callable[[str], int] = funcy.compose(
            int,
            operator.itemgetter(index_key),
            parse.compile(id_fmt).parse
        )
        # def _parse_index(self, elem_id) -> int:
        #     return int(parse.parse(self.id_fmt, elem_id)[self.index_key])

        def _format_index(index: int) -> str:
            return self.id_fmt.format(**{self.index_key: index})
        self._format_index: Callable[[int], str] = _format_index

        self.save_method: Optional[SaverFunction[T]] = save_method
        self.load_method: Optional[LoaderFunction[T]] = load_method
        self.creator: Optional[Creator[T]] = creator
        self.post_processor: Optional[Transformer[T, S]] = post_processor
        self.verbose: VerboseType = verbose

    def __getitem__(self, elem_id: str):
        self.load_lookup_table()
        return self._lookup_table.setdefault(self.keys.entries, {})[elem_id]

    def __setitem__(self, elem_id: str, entry: Mapping[str, Any]) -> None:
        self._lookup_table.setdefault(self.keys.entries, {})[elem_id] = entry
        self.save_lookup_table()

    def __delitem__(self, elem_id: str) -> None:
        del self._lookup_table.setdefault(self.keys.entries, {})[elem_id]
        self.save_lookup_table()

    @property
    def entries(self) -> dict:
        return dict(self._lookup_table[self.keys.entries])

    def __iter__(self) -> Iterator[str]:
        return self.entries.__iter__()

    def __len__(self) -> int:
        return self.entries.__len__()

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

    def entry_dir(self, elem_id: str) -> Path:
        return bl.utils.ensure_dir(self.entries_dir / elem_id)

    def elem_path(self, elem_id: str) -> Path:
        return bl.utils.ensure_parent(self.entry_dir(elem_id) / self.keys.elements)

    def elem_workspace(self, elem_id: str) -> Path:
        return bl.utils.ensure_dir(
            self.entry_dir(elem_id) / self.keys.workspace
        )

    def _initialize_lookup_table(self) -> None:
        self._lookup_table = {
            self.keys.entries: {}
        }
        self.save_lookup_table()

    def save_lookup_table(self) -> None:
        bl.io.save_json(self._lookup_table, self._table_path, cls=self._json_encoder)
        return self

    def load_lookup_table(
        self,
        raise_if_fails: bool = False
    ) -> None:
        if not self._table_path.is_file():
            self._initialize_lookup_table()
        try:
            self._lookup_table = bl.io.load_json(self._table_path, cls=self._json_decoder)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            if raise_if_fails:
                raise
            else:
                self._initialize_lookup_table()
                self.load_lookup_table()

    def save_elem(self, elem: T, path: PathType) -> None:
        if self.save_method is None:
            raise ValueError(
                'this Manager\'s *save_method* is not set.'
                'Define it in the Manager\'s initialization'
                'or by defining it as a property.')

        path = bl.utils.ensure_parent(path)
        self.save_method(elem, path)

    def load_elem(self, path: PathType) -> T:
        if self.load_method is None:
            raise ValueError(
                'this Manager\'s *load_method* is not set.'
                'Define it in the Manager\'s initialization'
                'or by defining it as a property.')

        path = bl.utils.ensure_resolved(path)
        return self.load_method(path)

    def create_elem(
            self,
            creator: Optional[Creator[T]],
            params: Pack
    ) -> T:
        if creator is None:
            creator = self.creator

        if creator is None:
            raise ValueError(
                'this *creator* is not set.'
                'Define it in the Manager\'s initialization'
                'define it as a property'
                'or pass as argument to function call.')

        return creator(params)

    def post_process_elem(
            self,
            post_processor: Optional[Transformer[T, S]],
            elem: T,
            params: Pack
    ) -> S:
        if post_processor is None:
            post_processor = self.post_processor

        if post_processor is None:
            raise ValueError(
                'this Manager\'s *post_processor* is not set.'
                'Define it in the Manager\'s initialization'
                'define it as a property'
                'or pass as argument to function call.')

        return post_processor(elem, params)

    def contents(self, elem_id: Optional[str] = None):
        if elem_id is None:
            return {
                elem_id: self.contents(elem_id)
                for elem_id in self.entries
            }
        else:
            return self.entries.get(elem_id, {}).get(self.keys.elements, {})

    def metadata(self, elem_id: Optional[str] = None):
        if elem_id is None:
            return {
                elem_id: self.metadata(elem_id)
                for elem_id in self.entries
            }
        else:
            return self.entries.get(elem_id, {}).get(self.keys.metadata, {})

    @overload
    def elem_creator(self, elem_id: None) -> dict: ...

    @overload
    def elem_creator(self, elem_id: str): ...

    def elem_creator(self, elem_id=None):
        if elem_id is None:
            return {
                elem_id: self.elem_creator(elem_id)
                for elem_id in self.entries
            }
        else:
            return self.contents(elem_id).get(self.keys.creator)

    @overload
    def creator_description(self, elem_id: None) -> dict: ...

    @overload
    def creator_description(self, elem_id: str): ...

    def creator_description(self, elem_id=None):
        if elem_id is None:
            return {
                elem_id: self.creator_description(elem_id)
                for elem_id in self.entries
            }
        else:
            return self.contents(elem_id).get(self.keys.creator_params, Pack())

    @overload
    def elem_post_processor(self, elem_id: None) -> dict: ...

    @overload
    def elem_post_processor(self, elem_id: str): ...

    def elem_post_processor(self, elem_id=None):
        if elem_id is None:
            return {
                elem_id: self.elem_post_processor(elem_id)
                for elem_id in self.entries
            }
        else:
            return self.contents(elem_id).get(self.keys.post_processor)

    @overload
    def post_processor_description(self, elem_id: None) -> dict: ...

    @overload
    def post_processor_description(self, elem_id: str): ...

    def post_processor_description(self, elem_id=None):
        if elem_id is None:
            return {
                elem_id: self.post_processor_description(elem_id)
                for elem_id in self.entries
            }
        else:
            return self.contents(elem_id).get(self.keys.post_processor_params, Pack())

    def new_elem_id(self) -> str:
        '''Return a elem id that does not exist yet
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

    def _resolve_name(
            self,
            key: str,
            contents: Optional[Mapping] = None,
            obj=None,
            default_obj=None
    ) -> str:
        if contents is not None and key in contents:
            return contents[key]
        elif hasattr(obj, 'name'):
            return obj.name
        elif hasattr(default_obj, 'name'):
            return default_obj.name
        else:
            raise ValueError(f'could not deduce name from ({key}, contents)=({obj},{contents})')

    def _resolve_contents(
            self,
            contents: Optional[Mapping] = None,
            creator: Optional[Creator[T]] = None,
            creator_description: Pack = Pack(),
            post_processor: Optional[Transformer[T, S]] = None,
            post_processor_description: Pack = Pack()
    ) -> Dict[str, Union[str, Pack]]:
        if contents is not None:
            return contents
        else:
            creator_name = self._resolve_name(
                self.keys.creator, contents=contents,
                obj=creator, default_obj=self.creator
            )

            post_processor_name = self._resolve_name(
                self.keys.post_processor, contents=contents,
                obj=post_processor, default_obj=self.post_processor
            )

            return {
                self.keys.creator: creator_name,
                self.keys.creator_params: [
                    list(creator_description.args),
                    dict(creator_description.kwargs)
                ],
                self.keys.post_processor: post_processor_name,
                self.keys.post_processor_params: [
                    list(post_processor_description.args),
                    dict(post_processor_description.kwargs)
                ],
            }

    def _resolve_metadata(
            self,
            metadata: Optional[Mapping] = None,
            path: Optional[PathType] = None
    ) -> Mapping:
        if metadata is None and path is None:
            raise ValueError(
                f'cannot resolve metadata with (metadata, path)=({metadata}, {path})')

        if metadata is None:
            return {
                self.keys.path: path
            }
        else:
            return metadata

    def _resolve_entry(
            self,
            contents: Optional[Mapping] = None,
            creator: Optional[Creator[T]] = None,
            creator_description: Optional[Pack] = None,
            post_processor: Optional[Transformer[T, S]] = None,
            post_processor_description: Optional[Pack] = None,
            metadata: Optional[Mapping] = None,
            path: Optional[PathType] = None
    ) -> Mapping:
        contents = self._resolve_contents(
            contents=contents,
            creator=creator, creator_description=creator_description,
            post_processor=post_processor, post_processor_description=post_processor_description
        )
        metadata = self._resolve_metadata(metadata=metadata, path=path)

        if path is None:
            raise TypeError(f'invalid elem path: {path}')

        return {
            self.keys.elements: contents,
            self.keys.metadata: metadata
        }

    def elem_id(
            self,
            contents: Optional[Mapping] = None,
            creator: Optional[Creator[T]] = None,
            creator_description: Optional[Pack] = None,
            post_processor: Optional[Transformer[T, S]] = None,
            post_processor_description: Optional[Pack] = None,
            missing_ok: bool = True
    ) -> str:
        contents = self._resolve_contents(
            contents=contents,
            creator=creator, creator_description=creator_description,
            post_processor=post_processor, post_processor_description=post_processor_description
        )

        candidates = bl.utils.extract_keys(
            self.contents(),
            value=contents,
            cmp=partial(bl.utils.json_equivalent, cls=self._json_encoder)
        )
        elem_id = mit.only(candidates, default=_sentinel)

        if elem_id is not _sentinel:
            return elem_id
        elif missing_ok:
            return self.new_elem_id()
        else:
            raise ValueError(
                f'could not find elem with the following contents: {contents}')

    def has_elem(
            self,
            contents: Optional[Mapping] = None,
            creator: Optional[Creator[T]] = None,
            creator_description: Optional[Pack] = None,
            post_processor: Optional[Transformer[T, S]] = None,
            post_processor_description: Optional[Pack] = None,
    ) -> bool:
        try:
            return self.elem_id(
                contents=contents,
                creator=creator,
                creator_description=creator_description,
                post_processor=post_processor,
                post_processor_description=post_processor_description,
                missing_ok=False
            ) in self.entries
        except ValueError:
            return False

    def _repeated_elems(self) -> Tuple[Dict[str, List[str]], List[str]]:
        all_contents = self.contents().items()

        repetition_dict = {
            elem_id: sorted(
                {
                    other_id
                    for other_id, other_contents in all_contents
                    if other_id != elem_id and bl.utils.json_equivalent(
                        contents, other_contents, cls=self._json_encoder)
                },
                key=self._parse_index
            )
            for elem_id, contents in all_contents
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

    def _retrieve_elem(
            self,
            path: PathType,
            raise_if_load_fails: bool
    ) -> Tuple[bool, T]:
        try:
            return True, self.load_elem(path)
        except tuple(getattr(self.load_method, 'expected_exceptions', (FileNotFoundError, OSError))):
            if self.verbose >= 1:
                print('Load failed')
            if raise_if_load_fails:
                raise
            return False, None

    def update_entries(
            self,
            updater: Callable[[str, Parameters], Any],
            elem_ids: Optional[Iterable[str]] = None,
            save: bool = True
    ) -> None:
        if elem_ids is None:
            elem_ids = self.entries

        self.load_lookup_table()

        entries = self.entries
        for elem_id in elem_ids:
            old_entry = copy.deepcopy(entries[elem_id])
            new_entry = updater(elem_id, old_entry)

            self[elem_id] = new_entry

        if save:
            self.save_lookup_table()

    def retrieve_elem(
            self,
            contents: Optional[Mapping] = None,
            creator: Optional[Creator[T]] = None,
            creator_description: Optional[Pack] = None,
            post_processor: Optional[Transformer[T, S]] = None,
            post_processor_description: Pack = Pack(),
            raise_if_load_fails: bool = False,
    ) -> Tuple[bool, T]:
        if not self.has_elem(
                contents=contents,
                creator=creator, creator_description=creator_description,
                post_processor=post_processor, post_processor_description=post_processor_description
        ):
            return False, None

        elem_id = self.provide_entry(
            contents=contents,
            creator=creator, creator_description=creator_description,
            post_processor=post_processor, post_processor_description=post_processor_description,
            include=False,
            missing_ok=False
        )

        path = self.elem_path(elem_id)

        return self._retrieve_elem(
            path=path,
            raise_if_load_fails=raise_if_load_fails
        )

    def retrieve_elems(
            self,
            entry_pred: Optional[Callable[[Mapping], bool]] = None
    ) -> Iterable[Tuple[str, T]]:
        if entry_pred is None:
            elems = self.entries.keys()
        else:
            elems = (
                elem_id
                for elem_id, entry in self.entries.items()
                if entry_pred(entry)
            )

        elems = (
            (elem_id, self.elem_path(elem_id))
            for elem_id in elems
        )

        elems = (
            (elem_id, self._load_elem(path, raise_if_load_fails=False))
            for elem_id, path in elems
        )

        elems = (
            (elem_id, elem)
            for elem_id, (success, elem) in elems
            if success
        )

        return elems

    def provide_entry(
            self,
            contents: Optional[Mapping] = None,
            creator: Optional[Creator[_ElemType]] = None,
            creator_description: Optional[Pack] = None,
            post_processor: Optional[Transformer[_ElemType, _PostProcessedElemType]] = None,
            post_processor_description: Optional[Pack] = None,
            include: bool = False,
            missing_ok: bool = False
    ) -> str:
        contents = self._resolve_contents(
            contents=contents,
            creator=creator, creator_description=creator_description,
            post_processor=post_processor, post_processor_description=post_processor_description
        )

        if self.verbose >= 2:
            print('Providing entry for contents:')
            pprint.pprint(contents)

        elem_id = self.elem_id(
            contents=contents,
            missing_ok=missing_ok
        )
        path = self.elem_path(elem_id)
        elem_rel_path = bl.utils.relative_path(self.entries_dir, path)

        entry = self._resolve_entry(
            contents=contents,
            path=elem_rel_path
        )

        if include:
            self[elem_id] = entry

        return elem_id

    def provide_elem(
            self,
            contents: Optional[Mapping] = None,
            creator: Optional[Creator[T]] = None,
            creator_description: Pack = Pack(),
            creator_params: Pack = Pack(),
            post_processor: Optional[Transformer[T, S]] = None,
            post_processor_description: Pack = Pack(),
            post_processor_params: Pack = Pack(),
            load: Union[bool, Callable[[], Tuple[bool, T]]] = False,
            save: Union[bool, Callable[[Union[T, S]], Any]] = False,
            raise_if_load_fails: bool = False,
    ) -> Union[T, S]:
        """Provide an element.

        If *post_processor* is omitted, will try to use this *Manager*'s default *post_processor*.
        If *None*, no post processing is done.
        """

        elem_id = self.provide_entry(
            contents=contents,
            creator=creator, creator_description=creator_description,
            post_processor=post_processor, post_processor_description=post_processor_description,
            include=True,
            missing_ok=True
        )

        if self.verbose:
            print('Providing element', elem_id)

        path = self.elem_path(elem_id)

        if self.verbose:
            print('Element', elem_id, 'assigned to', path)

        success = False
        if callable(load):
            if self.verbose:
                print('Trying to load', elem_id, 'using custom loader')
            success, elem = load(path)
        elif load:
            if self.verbose:
                print('Trying to load', elem_id, 'using default loader')
            success, elem = self._retrieve_elem(
                path=path, raise_if_load_fails=raise_if_load_fails)

        if not success:
            if self.verbose:
                print('Couldn\'t load', elem_id)
                print('Creating', elem_id)
            elem = self.create_elem(creator, creator_params)
            if callable(save):
                if self.verbose:
                    print('Saving', elem_id, 'using custom saver')
                save(elem, path)
            elif save:
                if self.verbose:
                    print('Saving', elem_id, 'using default saver')
                self.save_elem(elem, path)

        if post_processor is not None:
            if self.verbose:
                print('Post-processing', elem_id)
            elem = self.post_process_elem(post_processor, elem, post_processor_params)

        return elem
