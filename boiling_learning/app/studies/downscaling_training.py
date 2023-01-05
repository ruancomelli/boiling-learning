from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    fit_boiling_model,
)
from boiling_learning.app.training.common import (
    get_baseline_architecture,
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.app.training.evaluation import cached_model_evaluator
from boiling_learning.model.evaluate import UncertainValue
from boiling_learning.model.training import compile_model
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()
console = Console()


@app.command()
def boiling1d(
    direct: bool = typer.Option(..., '--direct/--indirect'),
    factors: list[int] = typer.Option(list(range(1, 7))),
) -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    case = boiling_cases()[0]
    evaluator = cached_model_evaluator('boiling1d')
    evaluations: list[tuple[int, str, UncertainValue]] = []

    table = Table(
        'Factor',
        'Training',
        'Validation',
        'Test',
        title='Downscaling analysis',
    )

    for factor in factors:
        preprocessors = default_boiling_preprocessors(
            direct_visualization=direct,
            downscale_factor=factor,
        )
        datasets = get_image_dataset(
            case(),
            transformers=preprocessors,
            experiment='boiling1d',
        )
        compiled_model = get_baseline_architecture(
            datasets,
            normalize_images=True,
            strategy=strategy,
        ) | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )

        fit_model = fit_boiling_model(
            compiled_model,
            datasets,
            get_baseline_fit_params(),
            target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
            strategy=strategy,
        )

        compiled_model = fit_model.architecture | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )
        evaluation = evaluator(compiled_model, datasets)

        evaluations.extend(
            (
                (factor, 'train', evaluation.training_metrics['MSE'].value),
                (
                    factor,
                    'train',
                    evaluation.training_metrics['MSE'].value
                    - evaluation.training_metrics['MSE'].lower,
                ),
                (
                    factor,
                    'train',
                    evaluation.training_metrics['MSE'].value
                    + evaluation.training_metrics['MSE'].upper,
                ),
                (factor, 'val', evaluation.validation_metrics['MSE'].value),
                (
                    factor,
                    'val',
                    evaluation.validation_metrics['MSE'].value
                    - evaluation.validation_metrics['MSE'].lower,
                ),
                (
                    factor,
                    'val',
                    evaluation.validation_metrics['MSE'].value
                    + evaluation.validation_metrics['MSE'].upper,
                ),
                (factor, 'test', evaluation.test_metrics['MSE'].value),
                (
                    factor,
                    'test',
                    evaluation.test_metrics['MSE'].value - evaluation.test_metrics['MSE'].lower,
                ),
                (
                    factor,
                    'test',
                    evaluation.test_metrics['MSE'].value + evaluation.test_metrics['MSE'].upper,
                ),
            )
        )

        table.add_row(
            f'{factor}',
            f'{evaluation.training_metrics["MSE"]}',
            f'{evaluation.validation_metrics["MSE"]}',
            f'{evaluation.test_metrics["MSE"]}',
        )

    console.print(table)

    plot_data = pd.DataFrame(evaluations, columns=['downscaling factor', 'subset', 'loss'])
    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.barplot(ax=ax, data=plot_data, x='downscaling factor', y='Loss', hue='subset')
    ax.set_xlabel('Downscaling factor')
    ax.set_ylabel('Loss')

    figure_path = resolve(
        _downscaling_training_study_path() / f"boiling1d-{'direct' if direct else 'indirect'}.png",
        parents=True,
    )
    f.savefig(str(figure_path))


@app.command()
def condensation(
    factors: list[int] = typer.Option(list(range(1, 10))),
) -> None:
    raise NotImplementedError


def _downscaling_training_study_path() -> Path:
    return resolve(studies_path() / 'downscaling-training', dir=True)
