from typing import Callable, Dict, Hashable, Optional, Tuple


class DispatchError(Exception):
    pass


class TableDispatcher:
    def __init__(self, default: Optional[Callable] = None):
        self._default: Optional[Callable] = default
        self._dispatch_table: Dict[Tuple[Hashable, ...], Callable] = {}

    def dispatch(self, *keys: Hashable):
        def _dispatcher(call: Callable) -> TableDispatcher:
            self._dispatch_table[keys] = call
            return self

        return _dispatcher

    def __call__(self, *keys: Hashable):
        try:
            return self._dispatch_table[keys]
        except KeyError as e:
            if self._default is not None:
                return self._default
            else:
                raise DispatchError(
                    'no default callable'
                    f' and no dispatch form was provided for keys "{keys}".'
                ) from e


table_dispatch = TableDispatcher
