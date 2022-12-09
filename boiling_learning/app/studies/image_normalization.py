import typer
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.training.boiling1d import get_pretrained_baseline_boiling_model

app = typer.Typer()
console = Console()


@app.command()
def boiling1d() -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    tables: list[Table] = []

    for normalize in (False, True):
        table = Table(
            'Visualization',
            'Validation MSE',
            'Test MSE',
            title='Normalized' if normalize else 'Non-normalized',
        )

        for direct_visualization in (False, True):
            model = get_pretrained_baseline_boiling_model(
                direct_visualization=direct_visualization,
                normalize_images=normalize,
                strategy=strategy,
            )

            table.add_row(
                'Direct' if direct_visualization else 'Indirect',
                str(model.validation_metrics['MSE']),
                str(model.test_metrics['MSE']),
            )

        tables.append(table)

    console.print(Columns(tables))


@app.command()
def condensation() -> None:
    raise NotImplementedError
