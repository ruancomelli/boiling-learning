from pathlib import Path

import modin.pandas as pd

from boiling_learning.descriptions import describe
from boiling_learning.utils.pathutils import PathLike, resolve


class ExperimentalData:
    def __init__(self, path: PathLike) -> None:
        self.path = resolve(path)

        if not self.path.is_file():
            raise ValueError(
                'data path is not a valid file. Please pass a valid one as input. '
                f'Got {self.path}'
            )

    def as_dataframe(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.path)
        except FileNotFoundError as e:
            raise ValueError(
                'data path is not a valid file. Please pass a valid one as input. '
                f'Got {self.path}'
            ) from e


@describe.instance(ExperimentalData)
def _describe_image_dataset(obj: ExperimentalData) -> Path:
    return describe(obj.path)
