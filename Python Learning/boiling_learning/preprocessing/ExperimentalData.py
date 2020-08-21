from pathlib import Path
from typing import Optional

import pandas as pd

from boiling_learning.utils import PathType
import boiling_learning.utils as bl_utils


class ExperimentalData:
    def __init__(
            self,
            path: Optional[PathType] = None,
            data_path: Optional[PathType] = None,
            description_path: Optional[PathType] = None
    ):
        if None not in {path, data_path} or all(x is None for x in (path, data_path)):
            raise ValueError('either path or data_path must be given as parameter, not both.')

        self.data_path: Path
        self.description_path: Optional[Path] = None
        if path is None:
            self.data_path = bl_utils.ensure_resolved(data_path)
            if description_path is not None:
                self.description_path = bl_utils.ensure_resolved(description_path)
        else:
            path = bl_utils.ensure_resolved(path)
            self.data_path = path / 'data.csv'
            self.description_path = path / 'description.md'

        if not self.data_path.is_file():
            raise ValueError(f'data path is not a valid file. Please pass a valid one as input. Got {self.data_path}')
        if self.description_path is not None and not self.description_path.is_file():
            self.description_path = None

    def as_dataframe(self) -> pd.DataFrame:
        if not self.data_path.is_file():
            raise ValueError(f'data path is not a valid file. Please pass a valid one as input. Got {self.data_path}')

        return pd.read_csv(self.data_path)
