import itertools

import typer
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.automl.autofit_dataset import best_baseline_boiling1d_model
from boiling_learning.app.cancellation import CancelledError
from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    get_baseline_model_size,
)
from boiling_learning.app.training.common import get_baseline_compile_params
from boiling_learning.app.training.evaluation import cached_model_evaluator
from boiling_learning.model.training import compile_model

app = typer.Typer()
console = Console()


@app.command()
def boiling1d() -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    raise CancelledError('This study is cancelled for not providing useful information.')

    model_evaluator = cached_model_evaluator('boiling1d')
    case = boiling_cases()[0]
    tables: list[Table] = []
    for direct, crop_mode in itertools.product((False, True), ('center', 'random')):
        table = Table(
            'Metric',
            'Training',
            'Validation',
            'Test',
            title=(
                'Automatic machine learning - '
                + ('direct' if direct else 'indirect')
                + f' visualization - {crop_mode} crop.'
            ),
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

        # hypermodel = autofit_dataset(
        #     datasets,
        #     target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        #     normalize_images=True,
        #     max_model_size=baseline_architecture_size,
        #     goal=None,
        #     experiment='boiling1d',
        #     strategy=strategy,
        # )

        # compiled_model = LazyDescribed.from_describable(hypermodel.best_model()) | compile_model(
        #     **get_baseline_compile_params(strategy=strategy),
        # )

        model = best_baseline_boiling1d_model(
            direct_visualization=direct,
            strategy=strategy,
            target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
            normalize_images=True,
            max_model_size=get_baseline_model_size(
                direct_visualization=direct,
                strategy=strategy,
            ),
        ) | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )

        evaluation = model_evaluator(model, datasets, measure_uncertainty=False)
        for metric in evaluation.metrics_names:
            table.add_row(
                metric,
                str(evaluation.training_metrics[metric]),
                str(evaluation.validation_metrics[metric]),
                str(evaluation.test_metrics[metric]),
            )

        tables.append(table)

    console.print(Columns(tables))


@app.command()
def condensation() -> None:
    raise NotImplementedError
