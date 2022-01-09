from typing import Any, Callable, Type, TypeGuard, TypeVar, overload

_Any = TypeVar('_Any')

def isa(*types: Type[_Any]) -> Callable[[Any], TypeGuard[_Any]]: ...
