from pathlib import Path
from typing import Callable

from tinydb.table import Table

from boiling_learning.io.storage import json_serialize
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.utils import (
    JSONDataType,
    PathLike,
    ensure_parent,
    ensure_resolved,
)

Allocator = Callable[[Pack], Path]


class TableAllocator:
    def __init__(
        self,
        path: PathLike,
        db: Table,
        serializer: Callable[[Pack], JSONDataType] = json_serialize,
    ) -> None:
        self.path: Path = ensure_resolved(path)
        self.db: Table = db
        self.serializer: Callable[[Pack], JSONDataType] = serializer

    def _doc_path(self, doc_id: int) -> Path:
        return ensure_parent(self.path / f'{doc_id}.json')

    def _provide(self, serialized: JSONDataType) -> int:
        for doc in self.db:
            if doc == serialized:
                return doc.doc_id
        return self.db.insert(serialized)

    def __call__(self, pack: Pack) -> Path:
        serialized: JSONDataType = self.serializer(pack)
        doc_id: int = self._provide(serialized)
        return self._doc_path(doc_id)
