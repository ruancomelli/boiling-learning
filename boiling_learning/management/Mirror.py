class Mirror:
    '''
    >>> params = Mirror({'propagate': True},
        shared='this will be shared between value and desc',
        value_only=Mirror.Fork(
            value='this will be present only in the value fork'
        ),
        desc_only=Mirror.Fork(
            desc='this will be present only in the desc fork'
        ),
        diff=Mirror.Fork(
            desc='this goes to desc',
            value='this goes to value'
        ),
        deep=Mirror({'propagate': True},
            also_value=Mirror.Fork(
                value='this also goes to value'
            ),
            another_diff=Mirror.Fork(
                desc='this also goes to desc',
                value='this also goes to value'
            ),
            also_shared='this goes to both'
        )
    )
    >>> assert params['desc']['shared'] == params['value']['shared']
    >>> assert params['value']['value_only'] == 'this will be present only in the value fork'
    >>> assert 'value_only' not in params['desc']
    >>> assert params['desc']['diff'] == 'this goes to desc'
    >>> assert params['value']['diff'] == 'this goes to value'
    >>> assert params['desc']['deep']['another_diff'] == 'this also goes to desc'
    >>> assert params['value']['deep']['another_diff'] == 'this also goes to value'
    '''

    class Fork(dict):
        pass

    def __init__(self, config_params, **kwargs):
        self.config = config_params
        self.contents = kwargs
        self.forks = {}
        self.forked = False
        self.default = {}
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
            if isinstance(real_value, Mirror.Fork) or (
                self.config.get('propagate', False)
                and isinstance(real_value, Mirror)
                and real_value.forked
            ):
                self.forked = True
                for splitter_key in real_value:
                    self.forks.setdefault(splitter_key, self.default.copy())[
                        key
                    ] = real_value[splitter_key]
            else:
                self.default[key] = real_value
                for v in self.forks.values():
                    v[key] = real_value
        if not self.forked:
            self.forks = self.contents

        return self
