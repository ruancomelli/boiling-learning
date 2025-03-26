from collections.abc import Callable
from numbers import Real


def round_to_multiple(
    number: Real,
    base: int,
    rounder: Callable[[Real], int] = round,
) -> int:
    return base * rounder(number / base)
