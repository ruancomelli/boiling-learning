from typing import Callable, Dict, Hashable, Optional, TypeVar

_Callable = TypeVar('_Callable', bound=Callable)


class DispatchError(Exception):
    pass


class TableDispatcher(Dict[Hashable, Callable]):
    def __init__(self, default: Optional[Callable] = None):
        super().__init__()
        self._default: Optional[Callable] = default

    def dispatch(self, key: Hashable):
        def _dispatcher(call: _Callable) -> _Callable:
            self[key] = call
            return call

        return _dispatcher

    def __missing__(self, key: Hashable) -> Callable:
        if self._default is None:
            raise DispatchError(
                'no default callable'
                f' and no dispatch form was provided for key "{key}".'
            )

        return self._default


table_dispatch = TableDispatcher
