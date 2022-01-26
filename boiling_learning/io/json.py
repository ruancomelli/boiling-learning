from __future__ import annotations

import json as _json
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

from classes import AssociatedType, Supports
from classes import typeclass as _typeclass
from typing_extensions import Protocol, TypedDict, runtime_checkable

from boiling_learning.utils.dataclasses import asdict, is_dataclass_instance
from boiling_learning.utils.frozendict import frozendict
from boiling_learning.utils.functional import P, Pack
from boiling_learning.utils.table_dispatch import table_dispatch
from boiling_learning.utils.utils import JSONDataType, PathLike, resolve

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


decode = table_dispatch()


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
@encode.instance(float)
@encode.instance(str)
def _encode_basics(instance: _BasicType) -> _BasicType:
    return instance


@decode.dispatch(None)
@decode.dispatch(bool)
@decode.dispatch(int)
@decode.dispatch(float)
@decode.dispatch(str)
def _decode(obj: _BasicType) -> _BasicType:
    return obj


@runtime_checkable
class HasJSONEncode(Protocol):
    def __json_encode__(self) -> JSONDataType:
        ...


@encode.instance(protocol=HasJSONEncode)
def _encode_has_encode(instance: HasJSONEncode) -> JSONDataType:
    return instance.__json_encode__()


@encode.instance(Path)
def _encode_Path(instance: Path) -> str:
    return str(instance)


@decode.dispatch(Path)
def _decode_Path(obj: str) -> Path:
    return Path(obj)


class _PackOfJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, Pack) and serialize.supports(instance.pair())


class PackOfJSONSerializable(
    Pack[Supports[JSONSerializable], Supports[JSONSerializable]],
    metaclass=_PackOfJSONSerializableMeta,
):
    ...


@encode.instance(delegate=PackOfJSONSerializable)
def _encode_Pack(instance: PackOfJSONSerializable) -> List[SerializedJSONObject]:
    return serialize(instance.pair())


@decode.dispatch(P)
@decode.dispatch(Pack)
def _decode_Pack(obj: List[SerializedJSONObject]) -> Pack:
    args, kwargs = deserialize(obj)
    return Pack(args, kwargs)


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


@decode.dispatch(list)
def _decode_list(obj: List[JSONDataType]) -> list:
    return list(map(deserialize, obj))


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


@decode.dispatch(dict)
def _decode_dict(obj: Dict[str, JSONDataType]) -> dict:
    return {key: deserialize(value) for key, value in obj.items()}


class _TupleOfJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, tuple) and all(serialize.supports(item) for item in instance)


class TupleOfJSONSerializable(
    Tuple[Supports[JSONSerializable], ...], metaclass=_TupleOfJSONSerializableMeta
):
    ...


@encode.instance(delegate=TupleOfJSONSerializable)
def _encode_tuple(instance: TupleOfJSONSerializable) -> List[SerializedJSONObject]:
    return list(map(serialize, instance))


@decode.dispatch(tuple)
def _decode_tuple(obj: List[JSONDataType]) -> tuple:
    return tuple(map(deserialize, obj))


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
    return serialize(dict(instance))


@decode.dispatch(frozendict)
def _decode_frozendict(obj: Dict[str, Any]) -> frozendict:
    return frozendict(deserialize(obj))


class DataclassOfJSONSerializableFieldsMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return is_dataclass_instance(instance) and serialize.supports(asdict(instance))


class DataclassOfJSONSerializableFields(
    metaclass=DataclassOfJSONSerializableFieldsMeta,
):
    ...


@encode.instance(delegate=DataclassOfJSONSerializableFields)
def _encode_dataclass(
    instance: DataclassOfJSONSerializableFields,
) -> Dict[str, SerializedJSONObject]:
    return serialize(asdict(instance))


@encode.instance(Fraction)
def _encode_fraction(instance: Fraction) -> List[int]:
    return serialize(instance.as_integer_ratio())


@decode.dispatch(Fraction)
def _decode_fraction(instance: List[int]) -> Fraction:
    numerator, denominator = instance
    return Fraction(deserialize(numerator), deserialize(denominator))


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
