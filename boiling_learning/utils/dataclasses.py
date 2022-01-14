from typing import Any, Callable, Mapping, Optional, TypeVar, Union

import dataclassy
import funcy
from dataclassy import dataclass
from dataclassy.dataclass import DataClass
from dataclassy.functions import as_dict as asdict
from dataclassy.functions import is_dataclass

__all__ = (
    'asdict',
    'dataclass',
    'is_dataclass',
    'is_dataclass_class',
    'is_dataclass_instance',
)

_T = TypeVar('_T')


def is_dataclass_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_dataclass(obj)


def is_dataclass_instance(obj: Any) -> bool:
    return not isinstance(obj, type) and is_dataclass(obj)


def dataclass_from_mapping(
    mapping: Mapping[str, Any],
    dataclass_factory: Callable[..., _T],
    key_map: Optional[Union[DataClass, Mapping[str, str]]] = None,
) -> _T:
    if not is_dataclass_class(dataclass_factory):
        raise ValueError('*dataclass_factory* must be a dataclass.')

    dataclass_field_names = frozenset(dataclassy.fields(dataclass_factory))

    if key_map is None:
        return dataclass_factory(**funcy.select_keys(dataclass_field_names, mapping))

    if is_dataclass_instance(key_map):
        key_map = dataclassy.as_dict(key_map)

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

    return dataclass_from_mapping(dataclassy.as_dict(obj), parent)
