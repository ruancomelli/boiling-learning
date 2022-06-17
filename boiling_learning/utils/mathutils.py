from typing import Callable


def round_to_multiple(number: int, base: int, rounder: Callable[[float], int] = round) -> int:
    return base * rounder(number / base)
