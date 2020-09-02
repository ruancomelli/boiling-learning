from typing import (
    Callable,
    Mapping,
    Optional
)

import boiling_learning.utils as bl_utils


class ElementCreator:
    def __init__(
        self,
        method: Callable,
        name: str,
        default_params: Optional[dict] = None,
        expand_params: bool = False,
        resolver: Optional[Callable[[dict, dict], dict]] = None
    ):
        self.default_params: dict = default_params if default_params is not None else {}
        self.expand_params: bool = expand_params
        self.method: Callable = method
        self.name: str = name

        def default_resolver(default_params, params):
            return bl_utils.merge_dicts(default_params, params)

        if resolver is None:
            self.resolver = default_resolver
        else:
            self.resolver = resolver

    def __call__(self, params: Optional[Mapping] = None):
        if params is None:
            params = {}

        resolved_params = self.resolve(params)

        if self.expand_params:
            return self.method(**resolved_params)
        else:
            return self.method(resolved_params)

    def resolve(self, params: Mapping):
        return self.resolver(self.default_params, params)
