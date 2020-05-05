class Mirror:
    class Fork(dict):
        pass
    
    def __init__(self, config_params, **kwargs):
        self.config = config_params
        self.contents = kwargs
        self.forks = dict()
        self.forked = False
        self.default = dict()
        self.split()
        
    def __str__(self):
        return f'Mirror(config={self.config}, forked={self.forked}, contents={self.contents}, default={self.default}, forks={self.forks})'
        
    def __iter__(self):
        return self.forks.__iter__()
    
    def __getitem__(self, key):
        return self.forks.__getitem__(key)
        
    def split(self):
        self.forked = False
        for key, real_value in self.contents.items():
            if (
                isinstance(real_value, Mirror.Fork)
                or (
                    self.config.get('propagate', False) 
                    and isinstance(real_value, Mirror)
                    and real_value.forked
                )
            ):
                self.forked = True
                for splitter_key in real_value:
                    self.forks.setdefault(splitter_key, self.default.copy())[key] = real_value[splitter_key]
            else:
                self.default[key] = real_value
                for v in self.forks.values():
                    v[key] = real_value
        if not self.forked:
            self.forks = self.contents
            
        return self