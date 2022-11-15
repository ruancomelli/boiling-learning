from typing import Any, TypeVar

_Any = TypeVar('_Any')
Pair = tuple[_Any, _Any]


def typename(obj: Any) -> str:
    return type(obj).__name__
