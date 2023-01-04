import itertools

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
from boiling_learning.app.training.evaluation import cached_model_evaluator
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

        compiled_model = LazyDescribed.from_describable(hypermodel.best_model()) | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )

        evaluation = model_evaluator(compiled_model, datasets)
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
