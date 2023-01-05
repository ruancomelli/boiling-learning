import typer
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.datasets.preprocessed.condensation import condensation_dataset

app = typer.Typer()
console = Console()


@app.command()
def boiling1d() -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    table = Table(
        'Dataset',
        'Training length',
        'Validation length',
        'Test length',
        title='Dataset sizes',
    )

    for index, datasets in enumerate(boiling_datasets(direct_visualization=False), start=1):
        train, val, test = datasets()

        table.add_row(
            f'Dataset {index}',
            str(len(train)),
            str(len(val)),
            str(len(test)),
        )

    console.print(table)


@app.command()
def condensation() -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    train, val, test = condensation_dataset(each=1)()
    console.print('Train set size:', len(train))
    console.print('Val set size:', len(val))
    console.print('Test set size:', len(test))
