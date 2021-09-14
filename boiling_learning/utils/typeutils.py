from typing import Any, Tuple, TypeVar

import typeguard

_T = TypeVar('_T')
Many = Tuple[_T, ...]


def typechecks(obj, type_) -> bool:
    try:
        typeguard.check_type('', obj, type_)
        return True
    except TypeError:
        return False


def typename(obj: Any) -> str:
    return type(obj).__name__
