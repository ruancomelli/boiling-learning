from pathlib import Path
from typing import Any, Callable

from tinydb import TinyDB
from tinydb.table import Table
from tinydb_smartcache import SmartCacheTable

from boiling_learning.io.json import serialize
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.utils import JSONDataType, PathLike, ensure_dir, ensure_parent, resolve

Allocator = Callable[[Pack[Any, Any]], Path]


class TableAllocator:
    def __init__(
        self,
        path: PathLike,
        db: Table,
        serializer: Callable[[Pack[Any, Any]], JSONDataType] = serialize,
    ) -> None:
        self.path: Path = resolve(path)
        self.db: Table = db
        self.serializer: Callable[[Pack[Any, Any]], JSONDataType] = serializer

    def _doc_path(self, doc_id: int) -> Path:
        return ensure_parent(self.path / f'{doc_id}.json')

    def _provide(self, serialized: JSONDataType) -> int:
        for doc in self.db:
            if doc == serialized:
                return doc.doc_id
        return self.db.insert(serialized)

    def __call__(self, pack: Pack[Any, Any]) -> Path:
        serialized: JSONDataType = self.serializer(pack)
        doc_id: int = self._provide(serialized)
        return self._doc_path(doc_id)


def default_table_allocator(root: PathLike) -> TableAllocator:
    root = ensure_dir(root)
    datapath = ensure_dir(root / 'data')
    dbpath = root / 'db.json'

    db = TinyDB(str(dbpath))
    db.table_class = SmartCacheTable

    return TableAllocator(datapath, db)
