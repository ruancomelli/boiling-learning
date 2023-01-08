from pathlib import Path

import pandas as pd
import rich
import typer
from rich.columns import Columns
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.bridged.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    DEFAULT_BOILING_OUTLIER_FILTER,
)
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.paths import studies_path
from boiling_learning.datasets.sliceable import targets
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()
console = rich.console.Console()


@app.command()
def boiling1d() -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    datasets = boiling_datasets(direct_visualization=False)

    tables: list[Table] = []
    for dataset_index, dataset in enumerate(datasets):
        table = Table(
            'Nominal power level [W]',
            'Heat flux level [W/cm^2]',
            title=f'Dataset {dataset_index}',
        )

        data = (
            pd.DataFrame(
                [
                    (target['nominal_power'], target[DEFAULT_BOILING_HEAT_FLUX_TARGET])
                    for subset in dataset()
                    for target in targets(subset.prefetch(1024))
                    if DEFAULT_BOILING_OUTLIER_FILTER()(None, target)
                ],
                columns=['nominal power', 'heat flux'],
            )
            .groupby(['nominal power'])
            .mean()
        )

        for _, nominal_power, heat_flux in data.itertuples():
            table.add_row(nominal_power, heat_flux)

        tables.append(table)

    console.print(Columns(tables))


@app.command()
def condensation(
    each: int = typer.Option(60),
    normalize: bool = typer.Option(...),
) -> None:
    raise NotImplementedError


def _heat_flix_levels_study_path() -> Path:
    return resolve(studies_path() / 'heat-flux-levels', dir=True)
