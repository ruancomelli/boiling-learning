from typing import Any


def typename(obj: Any) -> str:
    return type(obj).__name__
