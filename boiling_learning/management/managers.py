import enum
import json
import operator
import pprint
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import funcy
import more_itertools as mit
import parse
from dataclassy import dataclass

from boiling_learning.io.io import (
    BoolFlagged,
    BoolFlaggedLoaderFunction,
    SaverFunction,
    load_json,
    save_json,
)
from boiling_learning.io.json_encoders import GenericJSONDecoder, GenericJSONEncoder
from boiling_learning.preprocessing.transformers import Creator, Transformer
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.sentinels import EMPTY, Emptiable
from boiling_learning.utils.utils import (  # JSONDataType,; TODO: maybe using JSONDataType would be a good idea
    PathLike,
    SimpleRepr,
    VerboseType,
    extract_keys,
    json_equivalent,
    missing_ints,
    print_verbose,
    relative_path,
    resolve,
)

# TODO: check out <https://www.mlflow.org/docs/latest/tracking.html>
# TODO: maybe include "status" in metadata
# TODO: maybe include a comment in metadata
# TODO: improve this... there is a lot of repetition and buggy cases.
# ? Perhaps a better idea would be to have only one way to pass a description: through elem_id. No *contents*, just the id...
# TODO: standardize "post_processor": sometimes *None* means *None*, other times it means "get the default one"

_ElemType = TypeVar('_ElemType')
_PostProcessedElemType = TypeVar('_PostProcessedElemType')

_default_table_saver = partial(save_json, cls=GenericJSONEncoder)
_default_table_loader = partial(load_json, cls=GenericJSONDecoder)
_default_description_comparer = partial(
    json_equivalent, encoder=GenericJSONEncoder, decoder=GenericJSONDecoder
)


class Manager(
    SimpleRepr,
    Mapping[str, Dict[str, Any]],
    Generic[_ElemType, _PostProcessedElemType],
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

    class MultipleIdsHandler(enum.Enum):
        RAISE = enum.auto()
        KEEP_FIRST = enum.auto()
        KEEP_LAST = enum.auto()
        KEEP_FIRST_LOADED = enum.auto()
        KEEP_LAST_LOADED = enum.auto()

    # class Element:
    #     # TODO: finish this and replace Manager interface
    #     def __init__(
    #             self,
    #             elem_id: str,
    #             path: PathLike,
    #             load_method: BoolFlaggedLoaderFunction[_ElemType],
    #             save_method: SaverFunction[_ElemType]
    #     ) -> None:
    #         self._id: str = elem_id
    #         self._is_loaded: bool = False
    #         self._path: Path = resolve(path)
    #         self._value: Union[Sentinel, _ElemType, _PostProcessedElemType]

    #     @property
    #     def id(self) -> str:
    #         return self._id

    #     @property
    #     def is_loaded(self) -> bool:
    #         return self._is_loaded

    #     @property
    #     def path(self) -> Path:
    #         return self._path

    #     def load(self) -> bool:
    #         success, self._value = self.load_method(self.path)
    #         self._is_loaded = success
    #         return success

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
        key_names: Keys = Keys(),
        table_saver: Callable[[Dict[str, Any], Path], Any] = _default_table_saver,
        table_loader: Callable[[Path], Dict[str, Any]] = _default_table_loader,
        description_comparer: Callable[
            [Mapping[str, Any], Mapping[str, Any]], bool
        ] = _default_description_comparer,
    ) -> None:
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
        self._path: Path = resolve(path, dir=True)
        self._table_path: Path = self.path / 'lookup_table.json'
        self._entries_dir_path: Path = resolve(self.path / self.key_names.entries, dir=True)
        self._shared_dir_path: Path = resolve(self.path / 'shared', dir=True)
        self._table_saver = table_saver
        self._table_loader = table_loader
        self._description_comparer = description_comparer
        self._lookup_table: Optional[Dict[str, dict]] = None

        self.load_lookup_table()

        self.id_fmt: str = id_fmt
        self.index_key: str = index_key

        self._parse_index: Callable[[str], int] = funcy.rcompose(
            parse.compile(id_fmt).parse, operator.itemgetter(index_key), int
        )

        def _format_index(index: int) -> str:
            return self.id_fmt.format(**{self.index_key: index})

        self._format_index: Callable[[int], str] = _format_index

        self.save_method: Optional[SaverFunction[_ElemType]] = save_method
        self.load_method: Optional[BoolFlaggedLoaderFunction[_ElemType]] = load_method
        self.creator: Optional[Creator[_ElemType]] = creator
        self.post_processor: Optional[
            Transformer[_ElemType, _PostProcessedElemType]
        ] = post_processor
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
        return resolve(self.entries_dir / elem_id, dir=True)

    def elem_path(self, elem_id: str) -> Path:
        return resolve(self.entry_dir(elem_id) / self.key_names.elements, parents=True)

    def elem_workspace(self, elem_id: str) -> Path:
        return resolve(self.entry_dir(elem_id) / self.key_names.workspace, dir=True)

    def _initialize_lookup_table(self) -> None:
        self._lookup_table = {self.key_names.entries: {}}
        self.save_lookup_table()

    def save_lookup_table(self) -> None:
        self._table_saver(self._lookup_table, self._table_path)

    def load_lookup_table(self, raise_if_fails: bool = False) -> None:
        if not self._table_path.is_file():
            self._initialize_lookup_table()
        try:
            self._lookup_table = self._table_loader(self._table_path)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            if raise_if_fails:
                raise

            self._initialize_lookup_table()
            self.load_lookup_table()

    def save_elem(self, elem: _ElemType, path: PathLike) -> None:
        if self.save_method is None:
            raise ValueError(
                'this Manager\'s *save_method* is not set.'
                ' Define it in the Manager\'s initialization'
                ' or by defining it as a property.'
            )

        path = resolve(path, parents=True)
        self.save_method(elem, path)

    def load_elem(self, path: PathLike) -> BoolFlagged[_ElemType]:
        if self.load_method is None:
            raise ValueError(
                'this Manager\'s *load_method* is not set.'
                ' Define it in the Manager\'s initialization'
                ' or by defining it as a property.'
            )

        path = resolve(path)
        return self.load_method(path)

    def contents(self, elem_id: Optional[str] = None):
        if elem_id is None:
            return {elem_id: self.contents(elem_id) for elem_id in self.entries}
        else:
            return self.entries.get(elem_id, {}).get(self.key_names.elements, {})

    def new_elem_id(self) -> str:
        '''Return a elem id that does not exist yet'''
        indices = sorted(
            mit.map_except(
                self._parse_index,
                self.entries.keys(),
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
            )
        )
        if indices:
            missing_elems = missing_ints(indices)
            index = mit.first(missing_elems, indices[-1] + 1)
        else:
            index = 0

        return self._format_index(index)

    def _print(self, verbosity: VerboseType, *args, **kwargs) -> None:
        print_verbose(verbosity <= self.verbose, *args, **kwargs)

    def _pprint(self, verbosity: VerboseType, *args, **kwargs) -> None:
        if verbosity <= self.verbose:
            pprint.pprint(*args, **kwargs)

    def _resolve_name(
        self,
        key: str,
        contents: Optional[Mapping] = None,
        obj=None,
        default_obj=None,
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
        post_processor_description: Pack = Pack(),
    ) -> Dict[str, Union[str, Pack]]:
        if contents is not None:
            return contents

        creator_name = self._resolve_name(
            self.key_names.creator,
            contents=contents,
            obj=creator,
            default_obj=self.creator,
        )

        if post_processor is None:
            post_processor_name = None
        else:
            post_processor_name = self._resolve_name(
                self.key_names.post_processor,
                contents=contents,
                obj=post_processor,
                default_obj=self.post_processor,
            )

        return {
            self.key_names.creator: creator_name,
            self.key_names.creator_params: [
                list(creator_description.args),
                dict(creator_description.kwargs),
            ],
            self.key_names.post_processor: post_processor_name,
            self.key_names.post_processor_params: [
                list(post_processor_description.args),
                dict(post_processor_description.kwargs),
            ],
        }

    def _resolve_metadata(
        self,
        metadata: Optional[Mapping] = None,
        path: Optional[PathLike] = None,
    ) -> Mapping:
        if metadata is None and path is None:
            raise ValueError(f'cannot resolve metadata with (metadata, path)=({metadata}, {path})')

        if metadata is None:
            return {self.key_names.path: path}
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
        path: Optional[PathLike] = None,
    ) -> Mapping:
        contents = self._resolve_contents(
            contents=contents,
            creator=creator,
            creator_description=creator_description,
            post_processor=post_processor,
            post_processor_description=post_processor_description,
        )
        metadata = self._resolve_metadata(metadata=metadata, path=path)

        if path is None:
            raise TypeError(f'invalid elem path: {path}')

        return {
            self.key_names.elements: contents,
            self.key_names.metadata: metadata,
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
            Callable[[MultipleIdsHandler, Tuple[str]], str],
        ] = MultipleIdsHandler.RAISE,
    ) -> str:
        contents = self._resolve_contents(
            contents=contents,
            creator=creator,
            creator_description=creator_description,
            post_processor=post_processor,
            post_processor_description=post_processor_description,
        )

        elem_id_candidates = tuple(
            extract_keys(self.contents(), value=contents, cmp=self._description_comparer)
        )
        n_candidates = len(elem_id_candidates)

        if n_candidates == 0:
            if missing_ok:
                return self.new_elem_id()
            else:
                raise ValueError(f'Could not find elem with the following contents: {contents}')
        elif n_candidates == 1:
            return elem_id_candidates[0]
        else:
            return self._handle_multiple_ids(elem_id_candidates, multiple_ids_handler)

    def _handle_multiple_ids(
        self,
        elem_id_candidates: Iterable[str],
        multiple_ids_handler: Union[
            MultipleIdsHandler,
            Tuple[MultipleIdsHandler, Pack],
            Callable[[MultipleIdsHandler, Tuple[str]], str],
        ],
    ) -> str:
        elem_id_candidates = tuple(elem_id_candidates)

        if callable(multiple_ids_handler):
            return multiple_ids_handler(elem_id_candidates)

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
                        print('Removing file', path)  # DEBUG
                        # path.unlink()
                    else:
                        print('Removing dir', path)  # DEBUG
                        # rmdir(path, recursive=True, missing_ok=True, keep=False)
            if remove_entries:
                for id_to_remove in ids_to_remove:
                    print('Removing entry', id_to_remove)  # DEBUG
                    # del self[id_to_remove]

        if handler is self.MultipleIdsHandler.RAISE and len(elem_id_candidates) > 1:
            raise ValueError(
                f'Expected at most one item in iterable, but got {elem_id_candidates}'
            )
        elif handler in {
            self.MultipleIdsHandler.KEEP_FIRST,
            self.MultipleIdsHandler.KEEP_LAST,
        }:
            resolved_id = sorted(
                elem_id_candidates,
                reverse=handler is self.MultipleIdsHandler.KEEP_LAST,
            )[0]
            _remove(resolved_id, elem_id_candidates)
            return resolved_id
        elif handler in {
            self.MultipleIdsHandler.KEEP_FIRST_LOADED,
            self.MultipleIdsHandler.KEEP_LAST_LOADED,
        }:
            loader = parameters.kwargs.get(
                'loader',
                lambda path: self._load_elem(path, raise_if_load_fails=False),
            )
            loadable_candidates = map(
                lambda elem_id: (elem_id, loader(self.elem_path(elem_id))),
                elem_id_candidates,
            )
            loadable_candidates = (
                elem_id for elem_id, (success, _) in loadable_candidates if success
            )
            resolved_id = sorted(
                loadable_candidates,
                reverse=handler is self.MultipleIdsHandler.KEEP_LAST_LOADED,
            )[0]
            _remove(resolved_id, elem_id_candidates)
            return resolved_id

        raise ValueError('invalid *multiple_ids_handler* passed.')

    def _load_elem(self, path: PathLike, raise_if_load_fails: bool) -> BoolFlagged[_ElemType]:
        success, elem = self.load_elem(path)

        if raise_if_load_fails and not success:
            raise RuntimeError('loading failed with *raise_if_load_fails*.')
        else:
            return success, elem

    def retrieve_elems(
        self, entry_pred: Optional[Callable[[Mapping], bool]] = None
    ) -> Iterable[Tuple[str, _ElemType]]:
        if entry_pred is None:
            elems = self.entries.keys()
        else:
            elems = (elem_id for elem_id, entry in self.entries.items() if entry_pred(entry))

        elems = ((elem_id, self.elem_path(elem_id)) for elem_id in elems)

        elems = (
            (elem_id, self._load_elem(path, raise_if_load_fails=False)) for elem_id, path in elems
        )

        elems = ((elem_id, elem) for elem_id, (success, elem) in elems if success)

        return elems

    def provide_entry(
        self,
        contents: Optional[Mapping] = None,
        creator: Optional[Creator[_ElemType]] = None,
        creator_description: Optional[Pack] = None,
        post_processor: Optional[Transformer[_ElemType, _PostProcessedElemType]] = None,
        post_processor_description: Pack = Pack(),
        include: bool = False,
        missing_ok: bool = False,
    ) -> str:
        contents = self._resolve_contents(
            contents=contents,
            creator=creator,
            creator_description=creator_description,
            post_processor=post_processor,
            post_processor_description=post_processor_description,
        )

        self._print(2, 'Providing entry for contents:')
        self._pprint(2, contents)

        elem_id = self.elem_id(contents=contents, missing_ok=missing_ok)
        path = self.elem_path(elem_id)
        elem_rel_path = relative_path(self.entries_dir, path)

        entry = self._resolve_entry(contents=contents, path=elem_rel_path)

        if include:
            self[elem_id] = entry

        return elem_id

    def provide_elem(
        self,
        elem_id: Optional[str] = None,
        contents: Optional[Mapping] = None,
        creator: Emptiable[Creator[_ElemType]] = EMPTY,
        creator_description: Pack = Pack(),
        creator_params: Pack = Pack(),
        post_processor: Optional[
            Emptiable[Transformer[_ElemType, _PostProcessedElemType]]
        ] = EMPTY,
        post_processor_description: Pack = Pack(),
        post_processor_params: Pack = Pack(),
        load: Union[bool, BoolFlaggedLoaderFunction[_ElemType]] = False,
        save: Union[bool, Callable[[Union[_ElemType, _PostProcessedElemType]], Any]] = False,
        raise_if_load_fails: bool = False,
        reload_after_save: bool = False,
    ) -> Union[_ElemType, _PostProcessedElemType]:
        """Provide an element.

        If *post_processor* is *None*, will try to use this *Manager*'s default *post_processor*.
        """
        if creator is EMPTY:
            creator = self.creator
        if post_processor is EMPTY:
            post_processor = self.post_processor

        if elem_id is None:
            elem_id = self.provide_entry(
                contents=contents,
                creator=creator,
                creator_description=creator_description,
                post_processor=post_processor,
                post_processor_description=post_processor_description,
                include=True,
                missing_ok=True,
            )
        elif elem_id not in self:
            raise ValueError(f'passed a non-existing id explicitly: {elem_id}')

        self._print(1, 'Providing element', elem_id)

        if post_processor is not None:
            contents = self.contents(elem_id)
            base_contents = {
                **contents,
                self.key_names.post_processor: None,
                self.key_names.post_processor_params: [[], {}],
            }
            elem = self.provide_elem(
                contents=base_contents,
                creator_params=creator_params,
                post_processor=None,
                post_processor_description=Pack(),
                load=load,
                save=save,
                raise_if_load_fails=raise_if_load_fails,
                reload_after_save=reload_after_save,
            )

            self._print(1, 'Post-processing', elem_id)

            elem = post_processor(
                elem,
                *post_processor_params.args,
                **post_processor_params.kwargs,
            )
        else:
            must_load = load or callable(load)
            must_save = save or callable(save)

            def _load(
                elem_id: str, path: Path, raise_if_load_fails: bool
            ) -> BoolFlagged[_ElemType]:
                if callable(load):
                    self._print(1, 'Trying to load', elem_id, 'using custom loader')
                    success, elem = load(path)
                else:
                    self._print(1, 'Trying to load', elem_id, 'using default loader')
                    success, elem = self._load_elem(
                        path=path, raise_if_load_fails=raise_if_load_fails
                    )

                if not success and raise_if_load_fails:
                    raise ValueError(
                        f'failed to load element {elem_id} with flag *raise_if_load_fails*'
                    )

                return success, elem

            path = self.elem_path(elem_id)

            self._print(1, 'Element', elem_id, 'assigned to', path)

            success = False
            if must_load:
                success, elem = _load(elem_id, path, raise_if_load_fails=raise_if_load_fails)

            if not success:
                self._print(1, 'Couldn\'t load', elem_id)
                self._print(1, 'Creating', elem_id)

                elem = creator(creator_params)

                if must_save:
                    if callable(save):
                        self._print(1, 'Saving', elem_id, 'using custom saver')
                        save(elem, path)
                    else:
                        self._print(1, 'Saving', elem_id, 'using default saver')
                        self.save_elem(elem, path)

                    if reload_after_save:
                        self._print(1, 'Reloading', elem_id)
                        success, elem = _load(elem_id, path, raise_if_load_fails=True)

        return elem
