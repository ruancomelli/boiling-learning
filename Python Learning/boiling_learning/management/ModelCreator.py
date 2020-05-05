import boiling_learning.utils

class ModelCreator:
    def __init__(
        self,
        creator_method,
        creator_name,
        default_params=None,
        expand_params: bool = False,
        resolver=None,
    ):
        self.default_params = default_params if default_params is not None else {}
        self.expand_params = expand_params
        self.creator_method = creator_method
        self.creator_name = creator_name
        
        def default_resolver(default_params, params):
            return boiling_learning.utils.merge_dicts(default_params, params)
        if resolver is None:
            self.resolver = default_resolver
        else:
            self.resolver = resolver
        
    def __call__(self, params=None):
        if params is None:
            params = {}
        
        resolved_params = self.resolve(params)
        
        if self.expand_params:
            return self.creator_method(**resolved_params)
        else:
            return self.creator_method(resolved_params)
        
    def resolve(self, params):
        return self.resolver(self.default_params, params)