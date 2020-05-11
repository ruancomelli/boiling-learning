from collections.abc import Mapping

class Parameters(Mapping):
    # class Fork(dict):
    #     pass
    
    @staticmethod
    def get_set(d, key):
        return {k: d[k] for k in key}

    @staticmethod
    def get_list(d, key):
        d_ = d
        for k in key:
            d_ = d_[k]
        return d_
        
    @staticmethod
    def set_set(d, key, value):
        for k in key:
            d[k] = value
    
    @staticmethod
    def set_list(d, key, value):
        paths = [[]]
        for k in key:
            if isinstance(k, set):
                paths = [
                    sublist + [k_]
                    for k_ in k
                    for sublist in paths
                ]
            else:
                for p in paths:
                    p.append(k)
        for p in paths:
            d_ = d
            for p_ in p[:-1]:
                d_ = d_.setdefault(p_, {})
            d_[p[-1]] = value

    def __init__(self, config=None, params=None):
        
        def is_set(x):
            return isinstance(x, set)
        def is_list(x):
            return isinstance(x, list)
               
        if config is None:
            config = {
                'get': [
                    (is_set, Parameters.get_set),
                    (is_list, Parameters.get_list)
                ],
                'set': [
                    (is_set, Parameters.set_set),
                    (is_list, Parameters.set_list)
                ]
            }
        self.config = config
        
        if params is None:
            params = {}
        self.params = params
        
    def register_get_method(self, pred, method):
        self.config['get'].append(pred, method)
        
    def register_set_method(self, pred, method):
        self.config['set'].append(pred, method)
        
    def __getitem__(self, key):
        for pred, func in self.config['get']:
            if pred(key):
                return func(self.params, key)
        return self.params[key]
        
    def __setitem__(self, key, value):
        for pred, func in self.config['set']:
            if pred(key):
                func(self.params, key, value)
                break
        else:
            self.params[key] = value
        
    def __delitem__(self, key):
        self.params.__delitem__(key)
        
    def __iter__(self):
        return self.params.__iter__()
        
    def __len__(self):
        return self.params.__len__()
    
    # def fork(self, forker_classes=(Parameters,), forker_markers=(Parameters.Fork,), propagate=True):
    # 	forked = False
    # 	forks = dict()
    # 	default = dict()
      
    #     for key, real_value in self.items():
    #         if (
    #             any(isinstance(real_value, forker_marker) for forker_marker in forker_markers)
    #             or (
    #                 propagate
    #                 and any(isinstance(real_value, forker_class) for forker_class in forker_classes)
    #                 and real_value.forked
    #             )
    #         ):
    #             forked = True
    #             for splitter_key in real_value:
    #                 forks.setdefault(splitter_key, default.copy())[key] = real_value[splitter_key]
    #         else:
    #             default[key] = real_value
    #             for v in forks.values():
    #                 v[key] = real_value
    #     if not forked:
    #         forks = self
            
    #     return forks