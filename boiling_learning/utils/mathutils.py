from numbers import Real
from typing import Callable


def round_to_multiple(number: Real, base: int, rounder: Callable[[Real], int] = round) -> int:
    return base * rounder(number / base)
