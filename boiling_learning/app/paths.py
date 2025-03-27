from pathlib import Path

from boiling_learning.app.constants import masters_path


def data_path() -> Path:
    return masters_path() / "data"


def analyses_path() -> Path:
    return masters_path() / "analyses"


def shared_cache_path() -> Path:
    return analyses_path() / "cache"


def studies_path() -> Path:
    return analyses_path() / "studies"
