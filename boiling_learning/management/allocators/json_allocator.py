from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from classes import AssociatedType, Supports, typeclass
from tinydb import TinyDB
from tinydb.table import Table
from tinydb_smartcache import SmartCacheTable
from typing_extensions import final

from boiling_learning.io import json
from boiling_learning.utils.descriptions import describe
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


class JSONTableAllocator:
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
        [Pack[Supports[JSONDescribable[JSONDataType]], Supports[JSONDescribable[JSONDataType]]]],
        JSONDataType,
    ] = json_describe,
) -> JSONTableAllocator:
    root = ensure_dir(root)
    datapath = ensure_dir(root / 'data')
    dbpath = root / 'db.json'

    db = TinyDB(str(dbpath))
    db.table_class = SmartCacheTable

    return JSONTableAllocator(datapath, db, describer=describer)
