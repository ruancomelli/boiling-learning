from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.displaying import units
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    fit_boiling_model,
    get_baseline_boiling_architecture,
)
from boiling_learning.app.training.common import (
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.app.training.evaluation import cached_model_evaluator
from boiling_learning.model.training import compile_model
from boiling_learning.transforms import dataset_sampler
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()
console = Console()

FRACTIONS = tuple(Fraction(i + 1, 100) for i in range(10)) + tuple(
    Fraction(i + 1, 10) for i in range(10)
)


@app.command()
def boiling1d(
    direct: bool = typer.Option(..., '--direct/--indirect'),
) -> None:
    logger.info('Analyzing learning curve')

    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    datasets = baseline_boiling_dataset(direct_visualization=direct)

    table = Table(
        'Subsample',
        'Training loss',
        'Validation loss',
        'Test loss',
        title='Learning curve',
    )

    evaluator = cached_model_evaluator('boiling1d')

    losses: list[tuple[Fraction, float, str]] = []
    for fraction in FRACTIONS:
        subsampled = (
            datasets | dataset_sampler(count=fraction, subset='train')
            if fraction != 1
            else datasets
        )

        model = get_baseline_boiling_architecture(
            direct_visualization=direct,
            normalize_images=True,
            strategy=strategy,
        ) | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )

        fit_model = fit_boiling_model(
            model,
            subsampled,
            get_baseline_fit_params(),
            target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
            strategy=strategy,
        )

        compiled_model = fit_model.architecture | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )
        evaluation = evaluator(compiled_model, subsampled)

        table.add_row(
            f'{fraction} ({float(fraction):.0%})',
            f'{evaluation.training_metrics["MSE"]}',
            f'{evaluation.validation_metrics["MSE"]}',
            f'{evaluation.test_metrics["MSE"]}',
        )

        losses.extend(
            (
                (fraction, evaluation.training_metrics['MSE'], 'train'),
                (fraction, evaluation.validation_metrics['MSE'], 'val'),
                (fraction, evaluation.test_metrics['MSE'], 'test'),
            )
        )

    console.print(table)

    plot_data = pd.DataFrame(losses, columns=['fraction', 'loss', 'subset'])
    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.scatterplot(ax=ax, data=plot_data, x='fraction', y='loss', hue='subset')
    ax.set_xlabel('Dataset subsample size')
    ax.set_ylabel(f'Loss [{units["mse"]}]')
    ax.set_xscale('log')
    ax.set_yscale('log')

    f.savefig(_learning_curve_study_path() / f"boiling1d-{'direct' if direct else 'indirect'}.png")


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _learning_curve_study_path() -> Path:
    return resolve(studies_path() / 'learning-curve', dir=True)
