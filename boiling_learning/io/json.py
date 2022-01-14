from __future__ import annotations

import json as _json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

from classes import AssociatedType, Supports
from classes import typeclass as _typeclass
from frozendict import frozendict
from typing_extensions import Protocol, TypedDict, runtime_checkable

from boiling_learning.utils.functional import Pack
from boiling_learning.utils.table_dispatch import table_dispatch
from boiling_learning.utils.utils import JSONDataType, PathLike, resolve

_T = TypeVar('_T')
BasicTypes = Union[None, bool, int, str, float]
_BasicType = TypeVar('_BasicType', bound=BasicTypes)


class SerializedJSONObject(TypedDict):
    type: Optional[str]
    contents: JSONDataType


SerializedJSONDataType = Union[BasicTypes, List[BasicTypes], SerializedJSONObject]


class JSONEncodable(AssociatedType):
    ...


class JSONEncodableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return encode.supports(instance)


class SupportsJSONEncodable(Supports[JSONEncodable], metaclass=JSONEncodableMeta):
    ...


@_typeclass(JSONEncodable)
def encode(instance: Supports[JSONEncodable]) -> JSONDataType:
    '''Return a JSON encoding of an object.'''


class JSONSerializable(AssociatedType):
    ...


class SupportsJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return serialize.supports(instance)


class SupportsJSONSerializable(
    Supports[JSONSerializable],
    metaclass=SupportsJSONSerializableMeta,
):
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


@encode.instance(Path)
def _encode_Path(instance: Path) -> str:
    return str(instance)


class SerializedPack(TypedDict):
    args: SerializedJSONObject
    kwargs: SerializedJSONObject


@encode.instance(Pack)
def _encode_Pack(instance: Pack) -> SerializedPack:
    return {'args': serialize(instance.args), 'kwargs': serialize(instance.kwargs)}


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


class _TupleOfJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, tuple) and all(
            serialize.supports(instance) for instance in instance
        )


class TupleOfJSONSerializable(
    Tuple[str, Supports[JSONSerializable]], metaclass=_TupleOfJSONSerializableMeta
):
    ...


@encode.instance(delegate=TupleOfJSONSerializable)
def _encode_tuple(instance: TupleOfJSONSerializable) -> List[SerializedJSONObject]:
    return list(map(serialize, instance))


class _FrozenDictOfJSONEncodableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, frozendict) and all(
            isinstance(key, str) and encode.supports(value) for key, value in instance.items()
        )


class FrozenDictMeta(_FrozenDictOfJSONEncodableMeta, type(frozendict)):
    pass


class FrozenDictOfJSONEncodable(
    frozendict[str, Supports[JSONEncodable]],
    metaclass=FrozenDictMeta,
):
    ...


@encode.instance(delegate=FrozenDictOfJSONEncodable)
def _encode_frozendict(
    instance: FrozenDictOfJSONEncodable,
) -> Dict[str, SerializedJSONObject]:
    return encode(dict(instance))


decode = table_dispatch()


@decode.dispatch(None)
@decode.dispatch(bool)
@decode.dispatch(int)
@decode.dispatch(float)
@decode.dispatch(str)
def _decode(obj: _T) -> _T:
    return obj


@decode.dispatch(list)
def _decode_list(obj: List[JSONDataType]) -> list:
    return list(map(deserialize, obj))


@decode.dispatch(tuple)
def _decode_tuple(obj: List[JSONDataType]) -> tuple:
    return tuple(map(deserialize, obj))


@decode.dispatch(dict)
def _decode_dict(obj: Dict[str, JSONDataType]) -> dict:
    return {key: deserialize(value) for key, value in obj.items()}


@decode.dispatch(Pack)
def _decode_Pack(obj: SerializedPack) -> Pack:
    args = obj['args']
    kwargs = obj['kwargs']
    return Pack(deserialize(args), deserialize(kwargs))


@decode.dispatch(Path)
def _decode_Path(obj: str) -> Path:
    return Path(obj)


@serialize.instance(None)
@serialize.instance(bool)
@serialize.instance(int)
@serialize.instance(str)
@serialize.instance(float)
def _serialize_basics(instance: _BasicType) -> _BasicType:
    return instance


@serialize.instance(delegate=SupportsJSONEncodable)
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
