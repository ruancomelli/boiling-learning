from functools import partial
from typing import (
    Callable,
    Generic,
    Mapping,
    Optional,
    TypeVar
)

import boiling_learning.utils as bl_utils


T = TypeVar('T')


class ElementCreator(
        bl_utils.FrozenNamedMixin,
        Generic[T]
):
    # TODO: improve type annotation in Python 3.8+ using:
    #   - expand_params: Literal[True], method: Callable[..., Any]
    #   - expand_params: Literal[False], method: Callable[[Mapping], Any]
    # or something like this
    def __init__(
        self,
        name: str,
        method: Callable[..., T],
        default_params: Optional[dict] = None,
        expand_params: bool = False
    ):
        super().__init__(name)

        self.default_params: dict = default_params if default_params is not None else {}
        self.expand_params: bool = expand_params
        self.method: Callable[..., T] = method
        self.resolve = partial(bl_utils.merge_dicts, default_params)

    def __call__(self, params: Optional[Mapping] = None):
        if params is None:
            params = {}

        resolved_params = self.resolve(params)

        if self.expand_params:
            return self.method(**resolved_params)
        else:
            return self.method(resolved_params)


def make_creator(
        name: str,
        default_params: Optional[dict] = None,
        expand_params: bool = False,
        resolver: Optional[Callable[[dict, dict], dict]] = None
):
    def wrapper(method: Callable):
        return ElementCreator(
            method,
            name,
            default_params=default_params,
            expand_params=expand_params,
            resolver=resolver
        )
    return wrapper
