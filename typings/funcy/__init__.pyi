from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Hashable,
    Iterable,
    Mapping,
    Type,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import TypeGuard

_Mapping = TypeVar('_Mapping', bound=Mapping[Any, Any])
_Key = TypeVar('_Key', bound=Hashable)
_Any = TypeVar('_Any')
_Any1 = TypeVar('_Any1')
_Any2 = TypeVar('_Any2')
_Any3 = TypeVar('_Any3')
_Any4 = TypeVar('_Any4')

def identity(x: _Any) -> _Any: ...
def constantly(x: _Any) -> Callable[..., _Any]: ...
@overload
def isa(__type1: Type[_Any1]) -> Callable[[Any], TypeGuard[_Any1]]: ...
@overload
def isa(
    __type1: Type[_Any1], __type2: Type[_Any2]
) -> Callable[[Any], TypeGuard[Union[_Any1, _Any2]]]: ...
@overload
def isa(
    __type1: Type[_Any1], __type2: Type[_Any2], __type3: Type[_Any3]
) -> Callable[[Any], TypeGuard[Union[_Any1, _Any2, _Any3]]]: ...
@overload
def isa(
    __type1: Type[_Any1], __type2: Type[_Any2], __type3: Type[_Any3], __type4: Type[_Any4]
) -> Callable[[Any], TypeGuard[Union[_Any1, _Any2, _Any3, _Any4]]]: ...
@overload
def isa(*types: Type[_Any]) -> Callable[[Any], TypeGuard[_Any]]: ...
def omit(mapping: _Mapping, keys: Container[Any]) -> _Mapping: ...
def zipdict(keys: Iterable[_Key], vals: Iterable[_Any]) -> Dict[_Key, _Any]: ...
