from pathlib import Path
from typing import Any, Callable

from boiling_learning.utils.functional import Pack

Allocator = Callable[[Pack[Any, Any]], Path]
