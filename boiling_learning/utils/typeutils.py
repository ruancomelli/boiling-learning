from typing import Any

import typeguard


def typechecks(obj, type_) -> bool:
    try:
        typeguard.check_type('', obj, type_)
        return True
    except TypeError:
        return False


def typename(obj: Any) -> str:
    return type(obj).__name__
