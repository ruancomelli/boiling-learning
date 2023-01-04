from __future__ import annotations

from classes import AssociatedType, Supports, typeclass

from boiling_learning.model.evaluate import UncertainValue


class LaTeXEncodable(AssociatedType):
    ...


@typeclass(LaTeXEncodable)
def latexify(instance: Supports[LaTeXEncodable]) -> str:  # type: ignore[empty-body]
    '''Return a JSON encoding of an object.'''


@latexify.instance(UncertainValue)
def _latexify_uncertain_value(instance: UncertainValue) -> str:
    return f'\\uncertain{{{instance.value:f}}}{{{instance.upper:f}}}{{{instance.lower:f}}}'
