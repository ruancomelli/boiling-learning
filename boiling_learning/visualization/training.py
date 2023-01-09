from __future__ import annotations

import pandera as pa
from pandera.typing import DataFrame, Series

from boiling_learning.model.training import FitModelReturn


class TrainingHistory(pa.SchemaModel):
    epoch: Series[int] = pa.Field(ge=0)
    target: Series[float] = pa.Field()

    @classmethod
    def from_fit_model_return(
        cls,
        fit_model: FitModelReturn,
        /,
        *,
        epoch_key: str = 'epoch',
        target_key: str,
    ) -> DataFrame[TrainingHistory]:
        return DataFrame[TrainingHistory](
            {
                'epoch': entry[epoch_key],
                'target': entry[target_key],
            }
            for entry in fit_model.history
        )
