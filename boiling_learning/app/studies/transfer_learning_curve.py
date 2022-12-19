from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    fit_boiling_model,
    get_pretrained_baseline_boiling_model,
)
from boiling_learning.app.training.common import (
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.lazy import LazyDescribed
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

    datasets = boiling_datasets(direct_visualization=direct)[1]

    table = Table(
        'Subsample',
        'Validation loss',
        'Test loss',
        title='Learning curve',
    )

    validation_losses: list[float] = []

    for fraction in FRACTIONS:
        subsampled = (
            datasets | dataset_sampler(count=fraction, subset='train')
            if fraction != 1
            else datasets
        )

        pretrained_model = get_pretrained_baseline_boiling_model(
            direct_visualization=direct,
            normalize_images=True,
            strategy=strategy,
        )

        compiled_model = LazyDescribed.from_describable(
            pretrained_model.architecture
        ) | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )

        fit_model = fit_boiling_model(
            compiled_model,
            subsampled,
            get_baseline_fit_params(),
            target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
            strategy=strategy,
        )

        table.add_row(
            f'{fraction} ({float(fraction):.0%})',
            f'{fit_model.validation_metrics["MSE"]:.2f}',
            f'{fit_model.test_metrics["MSE"]:.2f}',
        )

        validation_losses.append(fit_model.validation_metrics['MSE'])

    console.print(table)

    sns.set_style('whitegrid')
    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(list(map(float, FRACTIONS)), validation_losses)
    ax.set_xticks(list(map(float, FRACTIONS)))
    ax.set_xticklabels(FRACTIONS)
    ax.set_xlabel('Dataset subsample')
    ax.set_ylabel('Validation loss')
    ax.set_xscale('log')
    ax.set_yscale('log')

    figure_path = resolve(
        _learning_curve_study_path() / f"boiling1d-{'direct' if direct else 'indirect'}.png",
        parents=True,
    )
    f.savefig(str(figure_path))


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _learning_curve_study_path() -> Path:
    return studies_path() / 'learning-curve'