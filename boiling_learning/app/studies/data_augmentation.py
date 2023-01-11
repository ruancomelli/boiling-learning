import rich
import typer
from rich.columns import Columns
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
console = rich.console.Console()


@app.command()
def boiling1d() -> None:
    """Validate current implementation against reference."""
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    case = boiling_cases()[0]

    tables: list[Table] = []
    for direct in False, True:
        table = Table(
            'Crop type',
            'Validation loss',
            'Test loss',
            title=f'Random augmentation analysis - {"direct" if direct else "indirect"}',
        )

        for crop_mode in 'center', 'random':
            preprocessors = default_boiling_preprocessors(
                direct_visualization=direct,
                crop_mode=crop_mode,
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
                str(crop_mode.title()),
                str(fit_model.validation_metrics['MSE']),
                str(fit_model.test_metrics['MSE']),
            )

        tables.append(table)

    console.print(Columns(tables))


@app.command()
def condensation(
    each: int = typer.Option(60),
    normalize: bool = typer.Option(...),
) -> None:
    raise NotImplementedError
