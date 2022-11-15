import abc
import json as _json
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar, final

from classes import AssociatedType, Supports, typeclass
from loguru import logger

from boiling_learning.descriptions import describe
from boiling_learning.io import json
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.pathutils import PathLike, resolve

# Ensure that all databases/tables will now use the smart query cache


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


@json_describe.instance(delegate=json.SupportsJSONEncodable)
def _json_describe_json_serializable(instance: json.SupportsJSONEncodable) -> json.JSONDataType:
    return json.serialize(instance)


class DescribableAsJSONDescribableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
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
        suffix: str = '.json',
    ) -> None:
        root = resolve(path, dir=True)
        self.path = resolve(root / 'data', dir=True)
        self.db_path = root / 'db.json'
        self._data: Optional[list[json.JSONDataType]] = None
        self.describer = describer
        self.suffix = suffix

    @property
    def data(self) -> list[json.JSONDataType]:
        if self._data is None:
            self._data = self._load_db()

        return self._data

    @data.setter
    def data(self, data: list[json.JSONDataType]) -> None:
        self._data = data
        self._save_db()

    def _doc_path(self, doc_id: int) -> Path:
        return self.path / f'{doc_id}{self.suffix}'

    def _provide(self, serialized: json.JSONDataType) -> int:
        try:
            return self.data.index(serialized)
        except ValueError:
            self.data = self.data + [serialized]
            return len(self.data) - 1

    def _load_db(self) -> list[json.JSONDataType]:
        try:
            with self.db_path.open('r', encoding='utf-8') as file:
                return _json.load(file)
        except FileNotFoundError:
            return []

    def _save_db(self) -> None:
        with self.db_path.open('w', encoding='utf-8') as file:
            _json.dump(self.data, file)

    def __call__(self, pack: Pack[Any, Any]) -> Path:
        logger.debug(f'Allocating path for args {pack}')

        args, kwargs = pack.pair()  # this normalizes `Pack` and `P`
        serialized = self.describer(Pack(args, kwargs))
        logger.debug(f'Described arguments: {serialized}')

        doc_id = self._provide(serialized)
        path = self._doc_path(doc_id)
        logger.debug(f'Allocated path is {path}')
        return path
