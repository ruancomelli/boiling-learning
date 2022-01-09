from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from classes import AssociatedType, Supports, typeclass
from tinydb import TinyDB
from tinydb.table import Table
from tinydb_smartcache import SmartCacheTable
from typing_extensions import final

from boiling_learning.io import json
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.utils import JSONDataType, PathLike, ensure_dir, ensure_parent, resolve

_JSONDescription = TypeVar('_JSONDescription', bound=JSONDataType)


@final
class JSONDescribable(AssociatedType[_JSONDescription]):
    ...


@typeclass(JSONDescribable)
def json_describe(instance: Supports[JSONDescribable[_JSONDescription]]) -> _JSONDescription:
    '''Return a JSON description of an object.'''


@json_describe.instance(delegate=json.SupportsJSONSerializable)
def _json_describe_json_serializable(instance: json.SupportsJSONSerializable) -> JSONDataType:
    return json.serialize(instance)


class PackOfJSONDescribableMeta(type(Pack)):
    def __instancecheck__(cls, instance: Any) -> bool:
        return (
            isinstance(instance, Pack)
            and json_describe.supports(instance.args)
            and json_describe.supports(instance.kwargs)
        )


class PackOfJSONDescribable(
    Pack[Supports[JSONDescribable[_JSONDescription]], Supports[JSONDescribable[_JSONDescription]]],
    Generic[_JSONDescription],
    metaclass=PackOfJSONDescribableMeta,
):
    ...


class JSONAllocator:
    def __init__(
        self,
        path: PathLike,
        db: Table,
        *,
        describer: Callable[[Pack[Any, Any]], JSONDataType] = json_describe,
    ) -> None:
        self.path: Path = resolve(path)
        self.db: Table = db
        self.describer: Callable[[Pack[Any, Any]], JSONDataType] = describer

    def _doc_path(self, doc_id: int) -> Path:
        return ensure_parent(self.path / f'{doc_id}.json')

    def _provide(self, serialized: JSONDataType) -> int:
        for doc in self.db:
            if doc == serialized:
                return doc.doc_id
        return self.db.insert(serialized)

    def __call__(self, pack: Pack[Any, Any]) -> Path:
        serialized: JSONDataType = self.describer(pack)
        doc_id: int = self._provide(serialized)
        return self._doc_path(doc_id)


def default_table_allocator(
    root: PathLike,
    *,
    describer: Callable[
        [PackOfJSONDescribable[_JSONDescription]], _JSONDescription
    ] = json_describe,
) -> JSONAllocator:
    root = ensure_dir(root)
    datapath = ensure_dir(root / 'data')
    dbpath = root / 'db.json'

    db = TinyDB(str(dbpath))
    db.table_class = SmartCacheTable

    return JSONAllocator(datapath, db, describer=describer)
