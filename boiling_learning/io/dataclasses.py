from __future__ import annotations

from dataclasses import dataclass as _dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Type, TypeVar

from boiling_learning.dataclasses import (
    DataClass,
    is_dataclass_class,
    is_dataclass_instance,
    shallow_asdict,
)
from boiling_learning.io import json
from boiling_learning.io.storage import deserialize, save, serialize

Metadata = json.JSONDataType
_DataClassType = TypeVar('_DataClassType', bound=Type[DataClass])
_AnyCallable = TypeVar('_AnyCallable', bound=Callable[..., Any])


class DataclassOfJSONEncodableFieldsMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return is_dataclass_instance(instance) and json.encode.supports(shallow_asdict(instance))


class DataclassOfJSONEncodableFields(
    metaclass=DataclassOfJSONEncodableFieldsMeta,
):
    ...


@json.encode.instance(delegate=DataclassOfJSONEncodableFields)
def _encode_dataclass(
    instance: DataclassOfJSONEncodableFields,
) -> dict[str, json.SerializedJSONObject]:
    return json.serialize(shallow_asdict(instance))


class DataclassOfSaveableFieldsMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return is_dataclass_instance(instance) and save.supports(shallow_asdict(instance))


class DataclassOfSaveableFields(metaclass=DataclassOfSaveableFieldsMeta):
    ...


@serialize.instance(delegate=DataclassOfSaveableFields)
def _serialize_dataclass(instance: DataclassOfSaveableFields, path: Path) -> Metadata:
    return serialize(shallow_asdict(instance), path)


def register_deserializer_for_dataclass(dataclass_type: _DataClassType) -> _DataClassType:
    @deserialize.dispatch(dataclass_type)
    def _deserialize(path: Path, metadata: Metadata) -> DataClass:
        fields = deserialize[dict](path, metadata)
        return dataclass_type(**fields)

    @json.decode.dispatch(dataclass_type)
    def _decode(path: Path) -> DataClass:
        fields = json.decode[dict](path)
        return dataclass_type(**fields)

    return dataclass_type


def _auto_register_deserializer_for_dataclass(dataclass_decorator: _AnyCallable) -> _AnyCallable:
    @wraps(dataclass_decorator)
    def _wrapped(*args, **kwargs) -> Any:
        decorated_result = dataclass_decorator(*args, **kwargs)

        if is_dataclass_class(decorated_result):
            # This is the case where we are directly wrapping a class, like
            #
            # @dataclass
            # class C: pass
            #
            # In this situation, directly register the class.
            return register_deserializer_for_dataclass(decorated_result)
        else:
            # This is the case where we are first passing arguments to `dataclass`
            # before actually decorating a class, like
            #
            # @dataclass(frozen=True)
            # class C: pass
            #
            # In this situation, recurse by instructing the new decorator to
            # automatically register dataclasses.
            return _auto_register_deserializer_for_dataclass(decorated_result)

    return _wrapped


dataclass = _auto_register_deserializer_for_dataclass(_dataclass)
