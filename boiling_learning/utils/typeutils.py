from typing import Any, Tuple, TypeVar

import typeguard
from typing_extensions import Protocol

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


class SupportsLessThan(Protocol):
    def __lt__(self, other: Any) -> bool:
        pass


SupportsLessThanT = TypeVar('SupportsLessThanT', bound=SupportsLessThan)
