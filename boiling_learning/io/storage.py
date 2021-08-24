from importlib import import_module
from typing import Any, Dict, List, TypeVar

from plum import Dispatcher

from boiling_learning.utils.functional import Pack
from boiling_learning.utils.table_dispatch import table_dispatch
from boiling_learning.utils.utils import JSONDataType

_T = TypeVar('_T')
dispatch = Dispatcher()


@dispatch
def json_encode(obj: None) -> None:
    # default implementation: return object unmodified
    return obj


@json_encode.dispatch
def json_encode(obj: int) -> int:
    return obj


@json_encode.dispatch
def json_encode(obj: float) -> float:
    return obj


@json_encode.dispatch
def json_encode(obj: str) -> str:
    return obj


@json_encode.dispatch
def json_encode(obj: list) -> list:
    return list(map(json_serialize, obj))


@json_encode.dispatch
def json_encode(obj: tuple) -> list:
    return list(map(json_serialize, obj))


@json_encode.dispatch
def json_encode(obj: dict) -> dict:
    return {key: json_serialize(value) for key, value in obj.items()}


@json_encode.dispatch
def json_encode(obj: Pack) -> list:
    return [json_serialize(obj.args), json_serialize(dict(obj.kwargs))]


json_decode = table_dispatch()


def json_decode(obj: Any) -> Any:
    # default implementation: return object unmodified
    return obj


@json_decode.dispatch(None)
@json_decode.dispatch(int)
@json_decode.dispatch(float)
@json_decode.dispatch(str)
def json_decode(obj: _T) -> _T:
    return obj


@json_decode.dispatch(list)
def json_decode(obj: List[JSONDataType]) -> list:
    return list(map(json_deserialize, obj))


@json_decode.dispatch(tuple)
def json_decode(obj: List[JSONDataType]) -> tuple:
    return tuple(map(json_deserialize, obj))


@json_decode.dispatch(Pack)
def json_decode(obj: JSONDataType) -> Pack:
    args, kwargs = obj
    return Pack(json_deserialize(args), json_deserialize(kwargs))


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
