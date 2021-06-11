from importlib import import_module
from typing import Any, Dict, TypeVar

from plum import Dispatcher

from boiling_learning.utils.table_dispatch import table_dispatch

_T = TypeVar('_T')
dispatch = Dispatcher()


@dispatch
def json_encode(obj: Any) -> Any:
    # default implementation: return object unmodified
    return obj


@table_dispatch
def json_decode(obj: Any) -> Any:
    # default implementation: return object unmodified
    return obj


def json_serialize(obj: Any) -> Dict[str, Any]:
    return {
        'module': type(obj).__module__,
        'type': type(obj).__qualname__,
        'contents': json_encode(obj),
    }


def json_deserialize(obj: Dict[str, Any]) -> Any:
    module = import_module(obj['module'])
    obj_type = getattr(module, obj['type'])
    return json_decode(obj_type)(obj['contents'])
