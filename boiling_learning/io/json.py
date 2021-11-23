from __future__ import annotations

import json as _json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

from classes import AssociatedType, Supports
from classes import typeclass as _typeclass
from typing_extensions import Protocol, TypedDict, runtime_checkable

from boiling_learning.utils.functional import Pack
from boiling_learning.utils.table_dispatch import table_dispatch
from boiling_learning.utils.utils import JSONDataType, PathLike, resolve

_T = TypeVar('_T')
BasicTypes = Union[None, bool, int, str, float]
_BasicType = TypeVar('_BasicType', bound=BasicTypes)
_JSONDataType = TypeVar('_JSONDataType', bound=JSONDataType)


class SerializedJSONObject(TypedDict):
    type: Optional[str]
    contents: JSONDataType


SerializedJSONDataType = Union[BasicTypes, List[BasicTypes], SerializedJSONObject]
_SerializedJSONDataType = TypeVar('_SerializedJSONDataType', bound=SerializedJSONDataType)


class JSONEncodable(AssociatedType):
    ...


@_typeclass(JSONEncodable)
def encode(instance: Supports[JSONEncodable]) -> JSONDataType:
    '''Return a JSON encoding of an object.'''


class JSONSerializable(AssociatedType):
    ...


@_typeclass(JSONSerializable)
def serialize(obj: Supports[JSONSerializable]) -> SerializedJSONObject:
    '''Return a JSON serialization of an object.'''


@encode.instance(None)
@encode.instance(bool)
@encode.instance(int)
@encode.instance(str)
@encode.instance(float)
def _encode_basics(instance: _BasicType) -> _BasicType:
    return instance


@runtime_checkable
class HasJSONEncode(Protocol):
    def __json_encode__(self) -> JSONDataType:
        ...


@encode.instance(protocol=HasJSONEncode)
def _encode_has_encode(instance: HasJSONEncode) -> JSONDataType:
    return instance.__json_encode__()


class _ListOfJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, list) and all(serialize.supports(item) for item in instance)


class ListOfJSONSerializable(
    List[Supports[JSONSerializable]], metaclass=_ListOfJSONSerializableMeta
):
    ...


@encode.instance(delegate=ListOfJSONSerializable)
def _encode_list(instance: ListOfJSONSerializable) -> List[JSONDataType]:
    return [serialize(item) for item in instance]


class _DictOfJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, dict) and all(
            isinstance(key, str) and serialize.supports(value) for key, value in instance.items()
        )


class DictOfJSONSerializable(
    Dict[str, Supports[JSONSerializable]], metaclass=_DictOfJSONSerializableMeta
):
    ...


@encode.instance(delegate=DictOfJSONSerializable)
def _encode_dict(instance: DictOfJSONSerializable) -> Dict[str, SerializedJSONObject]:
    return {key: serialize(value) for key, value in instance.items()}


decode = table_dispatch()


@decode.dispatch(None)
@decode.dispatch(bool)
@decode.dispatch(int)
@decode.dispatch(float)
@decode.dispatch(str)
def _json_decode(obj: _T) -> _T:
    return obj


@decode.dispatch(list)
def _json_decode_list(obj: List[JSONDataType]) -> list:
    return list(map(deserialize, obj))


@decode.dispatch(tuple)
def _json_decode_tuple(obj: List[JSONDataType]) -> tuple:
    return tuple(map(deserialize, obj))


@decode.dispatch(dict)
def _json_decode_dict(obj: Dict[str, JSONDataType]) -> dict:
    return {key: deserialize(value) for key, value in obj.items()}


@decode.dispatch(Pack)
def _json_decode_Pack(obj: List[JSONDataType]) -> Pack:
    args, kwargs = obj
    return Pack(deserialize(args), deserialize(kwargs))


@decode.dispatch(Path)
def _json_decode_Path(obj: str) -> Path:
    return Path(obj)


@serialize.instance(None)
@serialize.instance(bool)
@serialize.instance(int)
@serialize.instance(str)
@serialize.instance(float)
def _serialize_basics(instance: _BasicType) -> _BasicType:
    return instance


class JSONEncodableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return encode.supports(instance)


class IsJSONEncodable(Supports[JSONEncodable], metaclass=JSONEncodableMeta):
    ...


@serialize.instance(delegate=IsJSONEncodable)
def _serialize_json_encodable(obj: Supports[JSONEncodable]) -> SerializedJSONObject:
    '''Return a JSON serialization of an object.'''
    return {
        'type': f'{type(obj).__module__}.{type(obj).__qualname__}' if obj is not None else None,
        'contents': encode(obj),
    }


def dumps(obj: Supports[JSONSerializable]) -> str:
    return _json.dumps(serialize(obj))


def dump(obj: Supports[JSONSerializable], path: PathLike) -> None:
    serialized: SerializedJSONObject = serialize(obj)

    with resolve(path, parents=True).open('w', encoding='utf-8') as file:
        _json.dump(serialized, file, indent=4)


def deserialize(obj: SerializedJSONDataType) -> Any:
    if isinstance(obj, dict):
        obj_type = obj['type']

        if obj_type is not None:
            modulepath, typename = obj_type.rsplit('.', maxsplit=1)
            module = import_module(modulepath)
            obj_type = getattr(module, typename)

        return decode[obj_type](obj['contents'])
    elif isinstance(obj, list):
        return [deserialize(item) for item in obj]
    else:
        return obj


def load(path: PathLike) -> Any:
    with resolve(path, parents=True).open('r', encoding='utf-8') as file:
        obj = _json.load(file)

    return deserialize(obj)


def loads(contents: str) -> Any:
    return deserialize(_json.loads(contents))
