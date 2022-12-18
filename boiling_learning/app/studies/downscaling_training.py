import typer
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    fit_boiling_model,
)
from boiling_learning.app.training.common import (
    get_baseline_architecture,
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.model.training import compile_model

app = typer.Typer()
console = Console()


@app.command()
def boiling1d(
    direct: bool = typer.Option(..., '--direct/--indirect'),
    factors: list[int] = typer.Option(list(range(1, 10)) + list(range(10, 100, 10))),
) -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    case = boiling_cases()[0]

    table = Table(
        'Factor',
        'Validation loss',
        'Test loss',
        title='Downscaling analysis',
    )

    for factor in factors:
        preprocessors = default_boiling_preprocessors(
            direct_visualization=direct,
            downscale_factor=factor,
        )
        dataset = get_image_dataset(
            case(),
            transformers=preprocessors,
            experiment='boiling1d',
        )
        compiled_model = get_baseline_architecture(
            dataset,
            normalize_images=True,
            strategy=strategy,
        ) | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )

        fit_model = fit_boiling_model(
            compiled_model,
            dataset,
            get_baseline_fit_params(),
            target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
            strategy=strategy,
        )

        table.add_row(
            str(factor),
            str(fit_model.validation_metrics['MSE']),
            str(fit_model.test_metrics['MSE']),
        )

    console.print(table)


@app.command()
def condensation(
    factors: list[int] = typer.Option(list(range(1, 10))),
) -> None:
    raise NotImplementedError
