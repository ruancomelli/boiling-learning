from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple
)

import more_itertools as mit

from boiling_learning.utils.utils import (
    PathType,
    SimpleRepr,
    SimpleStr
)
import boiling_learning as bl


def apply_to_obj(
    tpl: Tuple[
        Any,
        str,
        Iterable,
        Mapping[str, Any]
    ]
):
    if len(tpl) != 4:
        raise ValueError(
            'expected a tuple in the format (obj, fname, args, kwargs)')

    obj, fname, args, kwargs = tpl
    return getattr(obj, fname)(*args, **kwargs)


def apply_to_f(
    tpl: Tuple[
        Callable,
        Iterable,
        Mapping[str, Any]
    ]
):
    if len(tpl) != 3:
        raise ValueError('expected a tuple in the format (f, args, kwargs)')

    f, args, kwargs = tpl
    return f(*args, **kwargs)


def distribute_iterable(
    keys: Sequence[Hashable],
    iterable: Iterable,
    assignments: Optional[Mapping[Hashable, Iterable]] = None,
    assign_pred: Optional[Callable[[Hashable], bool]] = None,
    assign_iterable: Optional[Iterable] = None
) -> Dict[Hashable, List]:
    if (assign_pred, assign_iterable).count(None) == 1:
        raise ValueError(
            'either both or none of assign_pred and assign_iterable must be passed as arguments.')

    if assign_iterable is not None:
        assignments = distribute_iterable(
            [key for key in keys if assign_pred(key)],
            assign_iterable,
            assignments
        )

    if assignments is None:
        n_keys = len(keys)

        return dict(
            zip(
                keys,
                map(list, mit.distribute(n_keys, iterable))
            )
        )
    else:
        distributed = bl.utils.merge_dicts(
            {k: [] for k in keys},
            {k: list(v) for k, v in assignments.items()}
        )
        distributed_items = sorted(
            distributed.items(),
            key=(lambda pair: len(pair[1]))
        )

        n_keys = len(distributed_items)
        level, pos = 0, 0
        for item in iterable:
            distributed_items[pos][1].append(item)
            pos += 1
            if pos == n_keys or len(distributed_items[pos][1]) > level:
                level += 1
                pos = 0
        return dict(distributed_items)


class UserPool(Sequence, SimpleRepr, SimpleStr):
    # See <https://stackoverflow.com/a/23665658/5811400>

    def __init__(
            self,
            workers: Iterable[Hashable],
            manager: Optional[Hashable] = None,
            current: Optional[Hashable] = None,
            server: Optional[Hashable] = None,
            workers_key: str = 'allowed_users',
            manager_key: str = 'manager',
            server_key: str = 'server',
            enabled: bool = True
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

    def __getitem__(self, key: int) -> Hashable:
        return self.workers.__getitem__(key)

    def __len__(self) -> int:
        return self.workers.__len__()

    def enable(self) -> None:
        self.is_enabled = True

    def disable(self) -> None:
        self.is_enabled = False

    @contextmanager
    def enabled(self) -> Iterator['UserPool']:
        prev_state = self.is_enabled
        self.enable()
        yield self
        self.is_enabled = prev_state

    @contextmanager
    def disabled(self) -> Iterator['UserPool']:
        prev_state = self.is_enabled
        self.disable()
        yield self
        self.is_enabled = prev_state

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, current: Hashable) -> None:
        if current not in self:
            raise ValueError(
                f'notebook user {current} is not expected. Allowed users are {self.workers}.')
        self._current = current

    @property
    def clients(self) -> List[Hashable]:
        return list(filter(lambda worker: worker != self.manager, self))

    @classmethod
    def from_json(
            cls,
            path: PathType,
            workers_key: str = 'allowed_users',
            manager_key: str = 'manager',
            server_key: str = 'server'
    ) -> 'UserPool':
        config = bl.io.load_json(path)

        return cls(
            workers=config[workers_key],
            manager=config.get(manager_key),
            server=config.get(server_key)
        )

    def to_json(self, path: PathType) -> None:
        obj = {
            self.workers_key: self.workers,
            self.manager_key: self.manager,
        }
        if self.server is not None:
            obj[self.server_key] = self.server
        bl.io.save_json(obj, path)

    def distribute_iterable(
            self,
            iterable: Iterable,
            assignments: Optional[Mapping[Hashable, Iterable]] = None,
            assign_pred: Optional[Callable[[Hashable], bool]] = None,
            assign_iterable: Optional[Iterable] = None
    ) -> Mapping[Hashable, Iterable]:
        if assignments is not None:
            user_diff = set(assignments.keys()) - set(self)
            if user_diff:
                raise ValueError(f'some users were not expected: {user_diff}')

        return distribute_iterable(
            self,
            iterable,
            assignments=assignments,
            assign_pred=assign_pred,
            assign_iterable=assign_iterable
        )

    def get_iterable(
            self,
            iterable: Iterable
    ) -> Iterable:
        if self.is_enabled:
            return self.distribute_iterable(iterable)[self.current]
        else:
            return iterable

    def is_manager(self) -> bool:
        return self.current == self.manager

    def is_client(self) -> bool:
        return self.current in self.clients

    def is_server(self) -> bool:
        return self.current == self.server
