from __future__ import annotations

import math

from classes import AssociatedType, Supports, typeclass

from boiling_learning.model.evaluate import UncertainValue

NEW_LINE_TOKEN = "\\\\"


class LaTeXEncodable(AssociatedType): ...


@typeclass(LaTeXEncodable)
def latexify(instance: Supports[LaTeXEncodable]) -> str:  # type: ignore[empty-body]
    """Return a JSON encoding of an object."""


@latexify.instance(UncertainValue)
def _latexify_uncertain_value(instance: UncertainValue) -> str:
    position_to_round = min(
        _position_of_most_significant_digit(instance.upper),
        _position_of_most_significant_digit(instance.lower),
    )

    rounded_mean = round(instance.value, position_to_round)
    rounded_upper = round(instance.upper, position_to_round)
    rounded_lower = round(instance.lower, position_to_round)

    rounded_mean_str = f"{rounded_mean:.{position_to_round}f}"
    rounded_upper_str = f"{rounded_upper:.{position_to_round}f}"
    rounded_lower_str = f"{rounded_lower:.{position_to_round}f}"

    return f"\\uncertain{{{rounded_mean_str}}}{{{rounded_upper_str}}}{{{rounded_lower_str}}}"


def _position_of_most_significant_digit(x: float, /) -> int:
    return -int(math.floor(math.log10(abs(x))))
