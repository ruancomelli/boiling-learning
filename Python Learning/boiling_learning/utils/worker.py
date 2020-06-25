from collections.abc import Sequence
from contextlib import contextmanager

import more_itertools as mit

import boiling_learning as bl

def apply_to_obj(tpl):
    if len(tpl) != 4:
        raise ValueError('expected a tuple in the format (obj, fname, args, kwargs)')
    
    obj, fname, args, kwargs = tpl
    return getattr(obj, fname)(*args, **kwargs)

def apply_to_f(tpl):
    if len(tpl) != 3:
        raise ValueError('expected a tuple in the format (f, args, kwargs)')
    
    f, args, kwargs = tpl
    return f(*args, **kwargs)

class UserPool(Sequence):
    # See <https://stackoverflow.com/a/23665658/5811400>
    
    def __init__(
        self,
        workers,
        manager=None,
        current=None,
        server=None,
        workers_key='allowed_users',
        manager_key='manager',
        server_key='server',
        enabled=True
    ):
        if manager is None:
            workers = mit.peekable(workers)
            manager = workers[0]
        self.manager = manager
        
        self.workers = sorted(workers)
        
        if current is not None:
            self.current = current
        self.server = server
            
        self.workers_key = workers_key
        self.manager_key = manager_key
        self.server_key = server_key
        self.is_enabled = enabled
        
    def __getitem__(self, key):
        return self.workers.__getitem__(key)
    
    def __len__(self):
        return self.workers.__len__()
    
    def enable(self):
        self.is_enabled = True
    
    def disable(self):
        self.is_enabled = False
        
    @contextmanager
    def enabled(self):
        prev_state = self.is_enabled
        self.enable()
        yield self
        self.is_enabled = prev_state
        
    @contextmanager
    def disabled(self):
        prev_state = self.is_enabled
        self.disable()
        yield self
        self.is_enabled = prev_state
        
    @property
    def current(self):
        return self._current
    @current.setter
    def current(self, current):
        if current not in self:
            raise ValueError(f'notebook user {current} is not expected. Allowed users are {self.workers}.')
        self._current = current
        
    @property
    def clients(self):
        return list(filter(lambda worker: worker != self.manager, self))
    
    @classmethod
    def from_json(
        cls,
        path,
        workers_key='allowed_users',
        manager_key='manager',
        server_key='server'
    ):
        config = bl.io.load_json(path)
        
        return cls(
            workers=config[workers_key],
            manager=config.get(manager_key, None),
            server=config.get(server_key, None)
        )
        
    def to_json(self, path):
        obj = {
            self.workers_key: self.workers,
            self.manager_key: self.manager,
        }
        if self.server is not None:
            obj[self.server_key] = self.server
        bl.io.save_json(obj, path)
        
    def distribute_iterable(self, iterable):
        n_workers = len(self)

        return dict(
            zip(
                self,
                mit.distribute(n_workers, iterable)
            )
        )

    def get_iterable(self, iterable):
        if self.is_enabled:
            return self.distribute_iterable(iterable)[self.current]
        else:
            return iterable
    
    def is_manager(self):
        return self.current == self.manager
    
    def is_client(self):
        return self.current in self.clients
    
    def is_server(self):
        return self.current == self.server
    