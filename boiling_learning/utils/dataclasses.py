from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Callable, Mapping, Optional, Type, TypeVar, Union

import funcy
from typing_extensions import TypeGuard

__all__ = (
    'asdict',
    'dataclass',
    'field',
    'fields',
    'is_dataclass',
    'is_dataclass_class',
    'is_dataclass_instance',
)

_T = TypeVar('_T')
DataClass = Any


def is_dataclass_class(obj: Any) -> TypeGuard[Type[DataClass]]:
    return isinstance(obj, type) and is_dataclass(obj)


def is_dataclass_instance(obj: Any) -> TypeGuard[DataClass]:
    return not isinstance(obj, type) and is_dataclass(obj)


def dataclass_from_mapping(
    mapping: Mapping[str, Any],
    dataclass_factory: Callable[..., _T],
    key_map: Optional[Union[DataClass, Mapping[str, str]]] = None,
) -> _T:
    if not is_dataclass_class(dataclass_factory):
        raise ValueError('*dataclass_factory* must be a dataclass.')

    dataclass_field_names = frozenset(fields(dataclass_factory))

    if key_map is None:
        return dataclass_factory(**funcy.select_keys(dataclass_field_names, mapping))

    if is_dataclass_instance(key_map):
        key_map = asdict(key_map)

    key_map = funcy.select_keys(dataclass_field_names, key_map)
    translator = {v: k for k, v in key_map.items()}.get
    mapping = {translator(key, key): value for key, value in mapping.items()}
    return dataclass_from_mapping(mapping, dataclass_factory)


def to_parent_dataclass(obj: DataClass, parent: Callable[..., DataClass]) -> DataClass:
    if not is_dataclass_class(parent):
        raise ValueError('*parent* must be a dataclass.')

    if not is_dataclass_instance(obj):
        raise ValueError('*obj* must be a dataclass instance.')

    if not isinstance(obj, parent):
        raise ValueError('*obj* must be an instance of *parent*.')

    return dataclass_from_mapping(asdict(obj), parent)
