class Mirror:
    class Split(dict):
        pass
    
    def __init__(self, config_params, **kwargs):
        self.config = config_params
        self.real = dict(**kwargs)
        self.mirror = dict()
        self.mirrored = False
        self.default = dict()
        self.prop = False
        self.split()
        
    def __iter__(self):
        return self.mirror.__iter__()
    
    def __getitem__(self, key):
        return self.mirror.__getitem__(key)
        
    def split(self):
        self.mirrored = False
        for key, real_value in self.real.items():
            if (
                isinstance(real_value, Mirror.Split)
                or (
                    self.config.get('propagate', False) 
                    and isinstance(real_value, Mirror)
                    and real_value.mirrored
                )
            ):
                self.mirrored = True
                for splitter_key in real_value:
                    if splitter_key not in self.mirror:
                        self.mirror[splitter_key] = self.default.copy()
                    self.mirror[splitter_key][key] = real_value[splitter_key]
            else:
                self.default[key] = real_value
                for v in self.mirror.values():
                    v[key] = real_value
        if not self.mirrored:
            self.mirror = self.real
            
        return self