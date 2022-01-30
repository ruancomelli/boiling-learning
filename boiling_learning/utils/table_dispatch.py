from typing import Callable, Dict, Optional, TypeVar, overload

from typing_extensions import Literal

from boiling_learning.utils.sentinels import EMPTY, Emptiable

_T = TypeVar('_T')
Predicate = Callable[[_T], bool]
_Callable = TypeVar('_Callable', bound=Callable)


class DispatchError(Exception):
    pass


class TableDispatcher(Dict[type, Callable]):
    def __init__(self, default: Optional[Callable] = None) -> None:
        super().__init__()
        self._default: Optional[Callable] = default
        self._predicates: Dict[Predicate[type], Callable] = {}

    @overload
    def dispatch(self, key: type, *, predicate: None = None) -> Callable[[_Callable], _Callable]:
        ...

    @overload
    def dispatch(
        self, key: Literal[EMPTY], *, predicate: Predicate[type]
    ) -> Callable[[_Callable], _Callable]:
        ...

    def dispatch(
        self, key: Emptiable[type] = EMPTY, *, predicate: Optional[Predicate[type]] = None
    ) -> Callable[[_Callable], _Callable]:
        if key is not EMPTY:
            return self._dispatch_by_key(key)
        else:
            return self._dispatch_by_predicate(predicate)

    def _dispatch_by_key(self, key: type) -> Callable[[_Callable], _Callable]:
        def _dispatcher(call: _Callable) -> _Callable:
            self[key] = call
            return call

        return _dispatcher

    def _dispatch_by_predicate(self, predicate: Predicate[_T]) -> Callable[[_Callable], _Callable]:
        def _predicate(obj: _T) -> bool:
            return predicate(obj)

        def _dispatcher(call: _Callable) -> _Callable:
            self._predicates[_predicate] = call
            return call

        return _dispatcher

    def __missing__(self, key: type) -> Callable:
        for predicate, call in self._predicates.items():
            if predicate(key):
                return call

        if self._default is None:
            raise DispatchError(
                'no default callable' f' and no dispatch form was provided for key "{key}".'
            )

        return self._default
