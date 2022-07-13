import abc
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from classes import AssociatedType, Supports, typeclass
from loguru import logger
from tinydb import TinyDB
from tinydb.table import Table
from tinydb_smartcache import SmartCacheTable
from typing_extensions import final

from boiling_learning.io import json
from boiling_learning.utils.descriptions import describe
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.pathutils import PathLike, resolve


class Allocator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, pack: Pack[Any, Any]) -> Path:
        pass

    def allocate(self, *args: Any, **kwargs: Any) -> Path:
        return self(Pack(args, kwargs))


_JSONDescription = TypeVar('_JSONDescription', bound=json.JSONDataType)


@final
class JSONDescribable(AssociatedType[_JSONDescription]):
    ...


@typeclass(JSONDescribable)
def json_describe(instance: Supports[JSONDescribable[_JSONDescription]]) -> _JSONDescription:
    '''Return a JSON description of an object.'''


@json_describe.instance(delegate=json.SupportsJSONSerializable)
def _json_describe_json_serializable(instance: json.SupportsJSONSerializable) -> json.JSONDataType:
    return json.serialize(instance)


class DescribableAsJSONDescribableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        if not describe.supports(instance):
            return False

        description = describe(instance)

        # If an object's description is equal to itself, then it is not the job of this
        # instance check to try to check if it is describable.
        # We then return `False` and let some other checker do the check.
        if description == instance:
            return False

        return json_describe.supports(description)


class DescribableAsJSONDescribable(
    Supports[JSONDescribable[_JSONDescription]],
    Generic[_JSONDescription],
    metaclass=DescribableAsJSONDescribableMeta,
):
    ...


@json_describe.instance(delegate=DescribableAsJSONDescribable)
def _json_describe_describable(
    instance: DescribableAsJSONDescribable[_JSONDescription],
) -> _JSONDescription:
    return json_describe(describe(instance))


class JSONTableAllocator(Allocator):
    def __init__(
        self,
        path: PathLike,
        db: Table,
        *,
        describer: Callable[[Pack[Any, Any]], json.JSONDataType] = json_describe,
    ) -> None:
        self.path: Path = resolve(path)
        self.db: Table = db
        self.describer: Callable[[Pack[Any, Any]], json.JSONDataType] = describer

    def _doc_path(self, doc_id: int) -> Path:
        return resolve(self.path / f'{doc_id}.json', parents=True)

    def _provide(self, serialized: json.JSONDataType) -> int:
        for doc in self.db:
            if doc == serialized:
                return doc.doc_id
        return self.db.insert(serialized)

    def __call__(self, pack: Pack[Any, Any]) -> Path:
        logger.debug(f'Allocating path for args {pack}')

        args, kwargs = pack.pair()  # this normalizes `Pack` and `P`
        serialized = self.describer(Pack(args, kwargs))
        logger.debug(f'Described arguments: {serialized}')

        doc_id: int = self._provide(serialized)
        path = self._doc_path(doc_id)
        logger.debug(f'Allocated path is {path}')
        return path


def default_table_allocator(
    root: PathLike,
    *,
    describer: Callable[
        [
            Pack[
                Supports[JSONDescribable[json.JSONDataType]],
                Supports[JSONDescribable[json.JSONDataType]],
            ]
        ],
        json.JSONDataType,
    ] = json_describe,
) -> JSONTableAllocator:
    root = resolve(root, dir=True)
    datapath = resolve(root / 'data', dir=True)
    dbpath = root / 'db.json'

    db = TinyDB(str(dbpath))
    db.table_class = SmartCacheTable

    return JSONTableAllocator(datapath, db, describer=describer)
