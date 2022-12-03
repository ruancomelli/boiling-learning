import os
from functools import cache
from pathlib import Path

from boiling_learning.utils.mathutils import round_to_multiple
from boiling_learning.utils.pathutils import resolve


@cache
def masters_path() -> Path:
    return resolve(os.environ['MASTERS_PATH'])


BOILING_BASELINE_BATCH_SIZE = round_to_multiple(200, base=8)
