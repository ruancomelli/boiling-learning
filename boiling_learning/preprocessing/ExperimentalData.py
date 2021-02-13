from pathlib import Path
from typing import (
    Optional
)

from frozendict import frozendict
import modin.pandas as pd

from boiling_learning.utils import PathLike
import boiling_learning.utils as bl_utils
from boiling_learning.utils import geometry
from boiling_learning.utils.units import unit_registry as ureg


SAMPLES = frozendict({
    1: geometry.Cylinder(
        length=6.5 * ureg.centimeter,
        diameter=0.51 * ureg.millimeter
    ),
    2: geometry.Cylinder(
        length=6.5 * ureg.centimeter,
        diameter=0.51 * ureg.millimeter
    ),
    3: geometry.Cylinder(
        length=6.5 * ureg.centimeter,
        diameter=0.25 * ureg.millimeter
    ),
    4: geometry.RectangularPrism(
        length=6.5 * ureg.centimeter,
        width=1/16 * ureg.inch,
        thickness=0.0031 * ureg.inch
    ),
    5: geometry.RectangularPrism(
        length=6.5 * ureg.centimeter,
        width=1/16 * ureg.inch,
        thickness=0.0031 * ureg.inch
    )
})


class ExperimentalData:
    def __init__(
            self,
            path: Optional[PathLike] = None,
            data_path: Optional[PathLike] = None,
            description_path: Optional[PathLike] = None
    ):
        if (path, data_path).count(None) != 1:
            raise ValueError('exactly one of path or data_path must be given as parameter.')

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

        df = pd.read_csv(self.data_path)

        return df
