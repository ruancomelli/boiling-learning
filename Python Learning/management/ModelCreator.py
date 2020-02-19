import utils

class ModelCreator:
    def __init__(
        self,
        creator_method,
        creator_name,
        default_params=None,
        expand_params: bool = False,
        resolver=None,
        splitter=None
    ):
        self.default_params = default_params if default_params is not None else {}
        self.expand_params = expand_params
        self.creator_method = creator_method
        self.creator_name = creator_name
        
        def default_resolver(default_params, params):
            return utils.merge_dicts(default_params, params)
        if resolver is None:
            self.resolver = default_resolver
        else:
            self.resolver = resolver
        
        def default_splitter(description, params, hidden_params):
            return (utils.merge_dicts(params, description), utils.merge_dicts(params, hidden_params))
            
        if splitter is None:
            self.splitter = default_splitter
        else:
            self.splitter = splitter
        
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
        
    def split(self, description, params, hidden_params):
        return self.splitter(description, params, hidden_params)