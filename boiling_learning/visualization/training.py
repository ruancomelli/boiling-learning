import matplotlib.pyplot as plt
import pandera as pa
from pandera.typing import DataFrame, Series


class TrainingHistory(pa.SchemaModel):
    epoch: Series[int] = pa.Field(ge=0)
    target: Series[float] = pa.Field()


@pa.check_types
def plot_training_history(history: DataFrame[TrainingHistory]) -> None:
    plt.plot(history, x='epoch', y='target')
