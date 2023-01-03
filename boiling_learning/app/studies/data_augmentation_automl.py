import typer
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.automl.autofit_dataset import autofit_dataset
from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    get_pretrained_baseline_boiling_model,
)
from boiling_learning.app.training.common import get_baseline_compile_params
from boiling_learning.app.training.evaluation import evaluate_boiling_model_with_dataset
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.training import compile_model

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

    case = boiling_cases()[0]

    tables: list[Table] = []
    for direct in False, True:
        table = Table(
            'Crop mode',
            'Validation loss',
            'Test loss',
            title=(
                'Automatic machine learning - '
                + ('direct' if direct else 'indirect')
                + ' visualization'
            ),
        )
        for crop_mode in 'center', 'random':
            baseline_fit_return = get_pretrained_baseline_boiling_model(
                direct_visualization=direct,
                normalize_images=True,
                strategy=strategy,
            )

            baseline_architecture_size = baseline_fit_return.architecture().count_parameters(
                trainable=True,
                non_trainable=False,
            )

            preprocessors = default_boiling_preprocessors(
                direct_visualization=direct,
                crop_mode=crop_mode,
            )
            datasets = get_image_dataset(
                case(),
                transformers=preprocessors,
                experiment='boiling1d',
            )

            hypermodel = autofit_dataset(
                datasets,
                target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
                normalize_images=True,
                max_model_size=baseline_architecture_size,
                goal=None,
                experiment='boiling1d',
                strategy=strategy,
            )

            compiled_model = LazyDescribed.from_describable(
                hypermodel.best_model()
            ) | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )

            _, validation_metrics, test_metrics = evaluate_boiling_model_with_dataset(
                compiled_model,
                datasets,
            )

            table.add_row(
                crop_mode,
                str(validation_metrics['MSE']),
                str(test_metrics['MSE']),
            )
            tables.append(table)

    console.print(Columns(tables))


@app.command()
def condensation() -> None:
    raise NotImplementedError
