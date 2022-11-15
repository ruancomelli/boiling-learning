import itertools
import typing
from typing import Iterable, TypeVar

import more_itertools as mit

_T = TypeVar('_T')


def unsort(iterable: Iterable[_T]) -> tuple[Iterable[int], Iterable[_T]]:
    peekable = mit.peekable(iterable)

    if not peekable:
        return (), ()

    sorted_items, sorters = mit.sort_together((peekable, itertools.count()))
    _, unsorters = mit.sort_together((sorters, itertools.count()))
    return (
        typing.cast(Iterable[int], unsorters),
        typing.cast(Iterable[_T], sorted_items),
    )
