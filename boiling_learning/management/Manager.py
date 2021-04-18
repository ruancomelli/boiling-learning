import copy
import enum
import json
import operator
import pprint
from functools import partial
from pathlib import Path
from typing import (Any, Callable, Dict, Generic, Iterable, Iterator, List,
                    Mapping, Optional, Tuple, TypeVar, Union)

import funcy
import more_itertools as mit
import parse
from dataclassy import dataclass
from typing_extensions import overload

import boiling_learning as bl
from boiling_learning.io.io import BoolFlaggedLoaderFunction, SaverFunction
from boiling_learning.preprocessing.transformers import Creator, Transformer
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.Parameters import Parameters
from boiling_learning.utils.utils import (  # JSONDataType, # TODO: maybe using JSONDataType would be good
    PathLike, VerboseType, _Sentinel)

# TODO: check out <https://www.mlflow.org/docs/latest/tracking.html>
# TODO: maybe include "status" in metadata
# TODO: maybe include a comment in metadata
# TODO: improve this... there is a lot of repetition and buggy cases.
# ? Perhaps a better would be to have only one way to pass a description: through elem_id. No *contents*, just the id...
# TODO: standardize "post_processor": sometimes *None* means *None*, other times it means "get the default one"


_sentinel = _Sentinel.get_instance()
_ElemType = TypeVar('_ElemType')
_PostProcessedElemType = TypeVar('_PostProcessedElemType')


class Manager(
        bl.utils.SimpleRepr,
        Mapping[str, dict],
        Generic[_ElemType, _PostProcessedElemType]
):
    @dataclass(frozen=True, kwargs=True)
    class Keys:
        entries: str = 'entries'
        elements: str = 'element'
        metadata: str = 'metadata'
        creator: str = 'creator'
        creator_params: str = 'creator_params'
        post_processor: str = 'post_processor'
        post_processor_params: str = 'post_processor_params'
        workspace: str = 'workspace'
        path: str = 'path'

    # @dataclass(frozen=True, kwargs=True)
    # class Entry:
    #     # TODO: here
    #     creator: str
    #     creator_params: PackType
    #     post_processor: str
    #     post_processor_params: PackType

    _default_table_saver = partial(bl.io.save_json, cls=bl.io.json_encoders.GenericJSONEncoder)
    _default_table_loader = partial(bl.io.load_json, cls=bl.io.json_encoders.GenericJSONDecoder)
    _default_description_comparer = partial(
        bl.utils.json_equivalent,
        encoder=bl.io.json_encoders.GenericJSONEncoder,
        decoder=bl.io.json_encoders.GenericJSONDecoder
    )

    class MultipleIdsHandler(enum.Enum):
        RAISE = enum.auto()
        KEEP_FIRST = enum.auto()
        KEEP_LAST = enum.auto()
        KEEP_FIRST_LOADED = enum.auto()
        KEEP_LAST_LOADED = enum.auto()

    class Element:
        # TODO: finish this and replace Manager interface
        def __init__(
                self,
                elem_id: str,
                path: PathLike,
                load_method: BoolFlaggedLoaderFunction[_ElemType],
                save_method: SaverFunction[_ElemType]
        ):
            self._id: str = elem_id
            self._is_loaded: bool = False
            self._path: Path = bl.utils.ensure_resolved(path)
            self._value: Union[_Sentinel, _ElemType, _PostProcessedElemType]

        @property
        def id(self) -> str:
            return self._id

        @property
        def is_loaded(self) -> bool:
            return self._is_loaded

        @property
        def path(self) -> Path:
            return self._path

        def load(self) -> bool:
            success, self._value = self.load_method(self.path)
            self._is_loaded = success
            return success

    def __init__(
            self,
            path: PathLike,
            id_fmt: str = '{index}.data',
            index_key: str = 'index',
            save_method: Optional[SaverFunction[_ElemType]] = None,
            load_method: Optional[BoolFlaggedLoaderFunction[_ElemType]] = None,
            creator: Optional[Creator[_ElemType]] = None,
            post_processor: Optional[Transformer[_ElemType, _PostProcessedElemType]] = None,
            verbose: VerboseType = False,
            load_table: bool = True,
            key_names: Keys = Keys(),
            table_saver: Callable[[dict, Path], Any] = _default_table_saver,
            table_loader: Callable[[Path], dict] = _default_table_loader,
            description_comparer: Callable[[Mapping, Mapping], bool] = _default_description_comparer
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
                   'model': { // key_names.elements: contents
                       'creator': *creator_name*, // key_names.creator
                       'creator_params': *creator_params* // key_names.creator_params
                   }
                   'metadata': *metadata_dict*  // key_names.metadata: metadata
                }
                ...
            }
        }
        ```
        '''
        self.key_names: self.Keys = key_names
        self._path: Path = bl.utils.ensure_dir(path)
        self._table_path: Path = self.path / 'lookup_table.json'
        self._entries_dir_path: Path = bl.utils.ensure_dir(
            self.path / self.key_names.entries
        )
        self._shared_dir_path: Path = bl.utils.ensure_dir(self.path / 'shared')
        self._table_saver = table_saver
        self._table_loader = table_loader
        self._description_comparer = description_comparer

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

        self.save_method: Optional[SaverFunction[_ElemType]] = save_method
        self.load_method: Optional[BoolFlaggedLoaderFunction[_ElemType]] = load_method
        self.creator: Optional[Creator[_ElemType]] = creator
        self.post_processor: Optional[Transformer[_ElemType, _PostProcessedElemType]] = post_processor
        self.verbose: VerboseType = verbose

    def __getitem__(self, elem_id: str):
        self.load_lookup_table()
        return self._lookup_table.setdefault(self.key_names.entries, {})[elem_id]

    def __setitem__(self, elem_id: str, entry: Mapping[str, Any]) -> None:
        self._lookup_table.setdefault(self.key_names.entries, {})[elem_id] = entry
        self.save_lookup_table()

    def __delitem__(self, elem_id: str) -> None:
        del self._lookup_table.setdefault(self.key_names.entries, {})[elem_id]
        self.save_lookup_table()

    @property
    def entries(self) -> dict:
        return dict(self._lookup_table[self.key_names.entries])

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

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
        return bl.utils.ensure_parent(self.entry_dir(elem_id) / self.key_names.elements)

    def elem_workspace(self, elem_id: str) -> Path:
        return bl.utils.ensure_dir(
            self.entry_dir(elem_id) / self.key_names.workspace
        )

    def _initialize_lookup_table(self) -> None:
        self._lookup_table = {
            self.key_names.entries: {}
        }
        self.save_lookup_table()

    def save_lookup_table(self) -> None:
        self._table_saver(self._lookup_table, self._table_path)

    def load_lookup_table(
        self,
        raise_if_fails: bool = False
    ) -> None:
        if not self._table_path.is_file():
            self._initialize_lookup_table()
        try:
            self._lookup_table = self._table_loader(self._table_path)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            if raise_if_fails:
                raise
            else:
                self._initialize_lookup_table()
                self.load_lookup_table()

    def save_elem(self, elem: _ElemType, path: PathLike) -> None:
        if self.save_method is None:
            raise ValueError(
                'this Manager\'s *save_method* is not set.'
                ' Define it in the Manager\'s initialization'
                ' or by defining it as a property.')


        path = bl.utils.ensure_parent(path)
        self.save_method(elem, path)

    def load_elem(self, path: PathLike) -> Tuple[bool, _ElemType]:
        if self.load_method is None:
            raise ValueError(
                'this Manager\'s *load_method* is not set.'
                ' Define it in the Manager\'s initialization'
                ' or by defining it as a property.')

        path = bl.utils.ensure_resolved(path)
        return self.load_method(path)

    def contents(self, elem_id: Optional[str] = None):
        if elem_id is None:
            return {
                elem_id: self.contents(elem_id)
                for elem_id in self.entries
            }
        else:
            return self.entries.get(elem_id, {}).get(self.key_names.elements, {})

    def metadata(self, elem_id: Optional[str] = None):
        if elem_id is None:
            return {
                elem_id: self.metadata(elem_id)
                for elem_id in self.entries
            }
        else:
            return self.entries.get(elem_id, {}).get(self.key_names.metadata, {})

    @overload
    def elem_creator(self, elem_id: None) -> dict:
        ...

    @overload
    def elem_creator(self, elem_id: str):
        ...

    def elem_creator(self, elem_id: Optional[str] = None):
        if elem_id is None:
            return {
                elem_id: self.elem_creator(elem_id)
                for elem_id in self.entries
            }
        else:
            return self.contents(elem_id).get(self.key_names.creator)

    @overload
    def creator_description(self, elem_id: None) -> dict:
        ...

    @overload
    def creator_description(self, elem_id: str):
        ...

    def creator_description(self, elem_id: Optional[str] = None):
        if elem_id is None:
            return {
                elem_id: self.creator_description(elem_id)
                for elem_id in self.entries
            }
        else:
            return self.contents(elem_id).get(self.key_names.creator_params, Pack())

    @overload
    def elem_post_processor(self, elem_id: None) -> dict:
        ...

    @overload
    def elem_post_processor(self, elem_id: str):
        ...

    def elem_post_processor(self, elem_id: Optional[str] = None):
        if elem_id is None:
            return {
                elem_id: self.elem_post_processor(elem_id)
                for elem_id in self.entries
            }
        else:
            return self.contents(elem_id).get(self.key_names.post_processor)

    @overload
    def post_processor_description(self, elem_id: None) -> dict:
        ...

    @overload
    def post_processor_description(self, elem_id: str):
        ...

    def post_processor_description(self, elem_id: Optional[str] = None):
        if elem_id is None:
            return {
                elem_id: self.post_processor_description(elem_id)
                for elem_id in self.entries
            }
        else:
            return self.contents(elem_id).get(self.key_names.post_processor_params, Pack())

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
            creator: Optional[Creator[_ElemType]] = None,
            creator_description: Pack = Pack(),
            post_processor: Optional[Transformer[_ElemType, _PostProcessedElemType]] = None,
            post_processor_description: Pack = Pack()
    ) -> Dict[str, Union[str, Pack]]:
        if contents is not None:
            return contents
        else:
            creator_name = self._resolve_name(
                self.key_names.creator, contents=contents,
                obj=creator, default_obj=self.creator
            )

            if post_processor is None:
                post_processor_name = None
            else:
                post_processor_name = self._resolve_name(
                    self.key_names.post_processor, contents=contents,
                    obj=post_processor, default_obj=self.post_processor
                )

            return {
                self.key_names.creator: creator_name,
                self.key_names.creator_params: [
                    list(creator_description.args),
                    dict(creator_description.kwargs)
                ],
                self.key_names.post_processor: post_processor_name,
                self.key_names.post_processor_params: [
                    list(post_processor_description.args),
                    dict(post_processor_description.kwargs)
                ],
            }

    def _resolve_metadata(
            self,
            metadata: Optional[Mapping] = None,
            path: Optional[PathLike] = None
    ) -> Mapping:
        if metadata is None and path is None:
            raise ValueError(
                f'cannot resolve metadata with (metadata, path)=({metadata}, {path})')

        if metadata is None:
            return {
                self.key_names.path: path
            }
        else:
            return metadata

    def _resolve_entry(
            self,
            contents: Optional[Mapping] = None,
            creator: Optional[Creator[_ElemType]] = None,
            creator_description: Optional[Pack] = None,
            post_processor: Optional[Transformer[_ElemType, _PostProcessedElemType]] = None,
            post_processor_description: Optional[Pack] = None,
            metadata: Optional[Mapping] = None,
            path: Optional[PathLike] = None
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
            self.key_names.elements: contents,
            self.key_names.metadata: metadata
        }

    def elem_id(
            self,
            contents: Optional[Mapping] = None,
            creator: Optional[Creator[_ElemType]] = None,
            creator_description: Optional[Pack] = None,
            post_processor: Optional[Transformer[_ElemType, _PostProcessedElemType]] = None,
            post_processor_description: Optional[Pack] = None,
            missing_ok: bool = True,
            multiple_ids_handler: Union[
                MultipleIdsHandler,
                Tuple[MultipleIdsHandler, Pack],
                Callable[[MultipleIdsHandler, Tuple[str]], str]
            ] = MultipleIdsHandler.RAISE
    ) -> str:
        contents = self._resolve_contents(
            contents=contents,
            creator=creator, creator_description=creator_description,
            post_processor=post_processor, post_processor_description=post_processor_description
        )

        elem_id_candidates = tuple(
            bl.utils.extract_keys(
                self.contents(),
                value=contents,
                cmp=self._description_comparer
            )
        )
        n_candidates = len(elem_id_candidates)

        if n_candidates == 0:
            if missing_ok:
                return self.new_elem_id()
            else:
                raise ValueError(
                    f'Could not find elem with the following contents: {contents}')
        elif n_candidates == 1:
            return elem_id_candidates[0]
        else:
            return self._handle_multiple_ids(
                elem_id_candidates,
                multiple_ids_handler
            )

    def _handle_multiple_ids(
            self,
            elem_id_candidates: Iterable[str],
            multiple_ids_handler: Union[
                MultipleIdsHandler,
                Tuple[MultipleIdsHandler, Pack],
                Callable[[MultipleIdsHandler, Tuple[str]], str]
            ]
    ) -> str:
        elem_id_candidates = tuple(elem_id_candidates)

        if callable(multiple_ids_handler):
            return multiple_ids_handler(elem_id_candidates)
        else:
            if isinstance(multiple_ids_handler, self.MultipleIdsHandler):
                handler, parameters = multiple_ids_handler, Pack()
            elif isinstance(multiple_ids_handler, tuple):
                handler, parameters = multiple_ids_handler

            remove_entries = parameters.kwargs.get('remove_entries', False)
            remove_files = parameters.kwargs.get('remove_files', False)

            def _remove(resolved_id, elem_id_candidates) -> None:
                ids_to_remove = tuple(set(elem_id_candidates) - {resolved_id})

                if remove_files:
                    for id_to_remove in ids_to_remove:
                        path = self.elem_path(id_to_remove)
                        if path.is_file():
                            print('Removing file', path) # DEBUG
                            # path.unlink()
                        else:
                            print('Removing dir', path) # DEBUG
                            # bl_utils.rmdir(path, recursive=True, missing_ok=True, keep=False)
                if remove_entries:
                    for id_to_remove in ids_to_remove:
                        print('Removing entry', id_to_remove) # DEBUG
                        # del self[id_to_remove]

            if handler is self.MultipleIdsHandler.RAISE and len(elem_id_candidates) > 1:
                raise ValueError(f'Expected at most one item in iterable, but got {elem_id_candidates}')
            elif handler in {self.MultipleIdsHandler.KEEP_FIRST, self.MultipleIdsHandler.KEEP_LAST}:
                resolved_id = sorted(
                    elem_id_candidates,
                    reverse=handler is self.MultipleIdsHandler.KEEP_LAST
                )[0]
                _remove(resolved_id, elem_id_candidates)
                return resolved_id
            elif handler in {self.MultipleIdsHandler.KEEP_FIRST_LOADED, self.MultipleIdsHandler.KEEP_LAST_LOADED}:
                loader = parameters.kwargs.get('loader', lambda path: self._load_elem(path, raise_if_load_fails=False))
                loadable_candidates = map(
                    lambda elem_id: (elem_id, loader(self.elem_path(elem_id))),
                    elem_id_candidates
                )
                loadable_candidates = (
                    elem_id
                    for elem_id, (success, _) in loadable_candidates
                    if success
                )
                resolved_id = sorted(
                    loadable_candidates,
                    reverse=handler is self.MultipleIdsHandler.KEEP_LAST_LOADED
                )[0]
                _remove(resolved_id, elem_id_candidates)
                return resolved_id

        raise ValueError('invalid *multiple_ids_handler* passed.')

    def _repeated_elems(self) -> Tuple[Dict[str, List[str]], List[str]]:
        all_contents = self.contents().items()

        repetition_dict = {
            elem_id: sorted(
                {
                    other_id
                    for other_id, other_contents in all_contents
                    if other_id != elem_id and self._description_comparer(contents, other_contents)
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

    def _load_elem(
            self,
            path: PathLike,
            raise_if_load_fails: bool
    ) -> Tuple[bool, _ElemType]:
        success, elem = self.load_elem(path)

        if raise_if_load_fails and not success:
            raise RuntimeError('loading failed with *raise_if_load_fails*.')
        else:
            return success, elem

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

    def retrieve_elems(
            self,
            entry_pred: Optional[Callable[[Mapping], bool]] = None
    ) -> Iterable[Tuple[str, _ElemType]]:
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
            post_processor_description: Pack = Pack(),
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
            elem_id: Optional[str] = None,
            contents: Optional[Mapping] = None,
            creator: Union[object, Creator[_ElemType]] = _sentinel,
            creator_description: Pack = Pack(),
            creator_params: Pack = Pack(),
            post_processor: Optional[Union[object, Transformer[_ElemType, _PostProcessedElemType]]] = _sentinel,
            post_processor_description: Pack = Pack(),
            post_processor_params: Pack = Pack(),
            load: Union[bool, Callable[[], Tuple[bool, _ElemType]]] = False,
            save: Union[bool, Callable[[Union[_ElemType, _PostProcessedElemType]], Any]] = False,
            raise_if_load_fails: bool = False,
            reload_after_save: bool = False
    ) -> Union[_ElemType, _PostProcessedElemType]:
        """Provide an element.

        If *post_processor* is *None*, will try to use this *Manager*'s default *post_processor*.
        """
        if creator is _sentinel:
            creator = self.creator
        if post_processor is _sentinel:
            post_processor = self.post_processor

        if elem_id is None:
            elem_id = self.provide_entry(
                contents=contents,
                creator=creator,
                creator_description=creator_description,
                post_processor=post_processor,
                post_processor_description=post_processor_description,
                include=True,
                missing_ok=True
            )
        elif elem_id not in self:
            raise ValueError(f'passed a non-existing id explicitly: {elem_id}')

        if self.verbose:
            print('Providing element', elem_id)

        if post_processor is not None:
            contents = self.contents(elem_id)
            base_contents = {
                **contents,
                self.key_names.post_processor: None,
                self.key_names.post_processor_params: [[], {}]
            }
            elem = self.provide_elem(
                contents=base_contents,
                creator_params=creator_params,
                post_processor=None,
                post_processor_description=Pack(),
                load=load,
                save=save,
                raise_if_load_fails=raise_if_load_fails,
                reload_after_save=reload_after_save
            )

            if self.verbose:
                print('Post-processing', elem_id)
            elem = post_processor(elem, *post_processor_params.args, **post_processor_params.kwargs)
        else:
            must_load = load or callable(load)
            must_save = save or callable(save)

            def _load(elem_id: str, path: Path, raise_if_load_fails: bool) -> _ElemType:
                if callable(load):
                    if self.verbose:
                        print('Trying to load', elem_id, 'using custom loader')
                    success, elem = load(path)
                else:
                    if self.verbose:
                        print('Trying to load', elem_id, 'using default loader')
                    success, elem = self._load_elem(
                        path=path, raise_if_load_fails=raise_if_load_fails)

                if not success and raise_if_load_fails:
                    raise ValueError(f'failed to load element {elem_id} with flag *raise_if_load_fails*')

                return success, elem

            path = self.elem_path(elem_id)

            if self.verbose:
                print('Element', elem_id, 'assigned to', path)

            success = False
            if must_load:
                success, elem = _load(elem_id, path, raise_if_load_fails=raise_if_load_fails)

            if not success:
                if self.verbose:
                    print('Couldn\'t load', elem_id)
                    print('Creating', elem_id)
                elem = creator(creator_params)

                if must_save:
                    if callable(save):
                        if self.verbose:
                            print('Saving', elem_id, 'using custom saver')
                        save(elem, path)
                    else:
                        if self.verbose:
                            print('Saving', elem_id, 'using default saver')
                        self.save_elem(elem, path)

                    if reload_after_save:
                        if self.verbose:
                            print('Reloading', elem_id)
                        success, elem = _load(elem_id, path, raise_if_load_fails=True)

        return elem
