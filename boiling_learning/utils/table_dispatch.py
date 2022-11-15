from typing import Any, Callable, Optional, TypeVar

_Callable = TypeVar('_Callable', bound=Callable[..., Any])


class DispatchError(Exception):
    pass


class TableDispatcher(dict[type, Callable[..., Any]]):
    def __init__(self, default: Optional[Callable[..., Any]] = None) -> None:
        super().__init__()
        self._default: Optional[Callable[..., Any]] = default

    def dispatch(self, key: type) -> Callable[[_Callable], _Callable]:
        def _dispatcher(call: _Callable) -> _Callable:
            self[key] = call
            return call

        return _dispatcher

    def __missing__(self, key: type) -> Callable[..., Any]:
        if self._default is None:
            raise DispatchError(
                f'no default callable and no dispatch form was provided for key "{key}".'
            )

        return self._default
