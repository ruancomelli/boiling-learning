from typing import Iterable, Iterator, Tuple, TypeVar

_T = TypeVar('_T')


def flaglast(iterable: Iterable[_T]) -> Iterator[Tuple[bool, _T]]:
    it = iter(iterable)

    try:
        last = next(it)
    except StopIteration:
        return

    for val in it:
        yield False, last
        last = val

    yield True, last
