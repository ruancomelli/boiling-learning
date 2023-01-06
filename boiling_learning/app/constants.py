import os
from functools import cache
from pathlib import Path

from boiling_learning.utils.mathutils import round_to_multiple
from boiling_learning.utils.pathutils import resolve


@cache
def masters_path() -> Path:
    return resolve(os.environ['MASTERS_PATH'])


@cache
def figures_path() -> Path:
    return resolve(os.environ['FIGURES_PATH'])


BOILING_BASELINE_BATCH_SIZE = round_to_multiple(200, base=8)
BASELINE_BOILING_MSE_DIRECT = 13  # W / cm**2
BASELINE_BOILING_MSE_INDIRECT = 33  # W / cm**2

DEFAULT_CONDENSATION_MASS_RATE_TARGET = 'mass_rate'
