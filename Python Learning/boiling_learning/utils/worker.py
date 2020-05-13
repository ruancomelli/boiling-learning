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

class UserPool:
    # See <https://stackoverflow.com/a/23665658/5811400>
    
    def __init__(self, workers, manager=None, current=None, server=None):
        self.workers = workers
        
        if manager is None:
            manager = workers[0]
        self.manager = manager
        
        if current is not None:
            self.current = current
            
        if server is not None:
            self.server = server
        
    def __len__(self):
        return self.workers.__len__()
    
    @property
    def current(self):
        return self._current
    @current.setter
    def current(self, current):
        if current not in self.workers:
            raise ValueError(f'notebook user {current} is not expected. Allowed users are {self.workers}.')
        self._current = current
        
    @property
    def clients(self):
        return list(filter(lambda worker: worker != self.manager, self.workers))
    
    @staticmethod
    def from_json(path):
        config = bl.io.load_json(path)
        
        return UserPool(
            workers=config['allowed_users'],
            manager=config.get('manager', None),
            server=config.get('server', None)
        )
        
    def to_json(self, path):
        obj = {
            'allowed_users': self.workers,
            'manager': self.manager,
        }
        bl.io.save_json(obj, path)
        
    def _distribute_iterable(self, iterable):
        n_workers = len(self)

        return dict(
            zip(
                self.workers,
                mit.distribute(n_workers, iterable)
            )
        )

    def get_iterable(self, iterable):
        return self._distribute_iterable(iterable)[self.current]
    
    def is_manager(self):
        return self.current == self.manager
    
    def is_server(self):
        return self.current == self.server
    