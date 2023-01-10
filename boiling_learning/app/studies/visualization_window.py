from fractions import Fraction
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
from boiling_learning.model.training import compile_model
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()
console = Console()

FRACTIONS = tuple(Fraction(num, 100) for num in range(100, 9, -10))


@app.command()
def boiling1d() -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    for direct in False, True:
        direct_label = 'direct' if direct else 'indirect'

        table = Table(
            'Window size',
            'Training loss',
            'Validation loss',
            'Test loss',
            title=f'Visualization window analysis - {direct_label}',
        )

        case = boiling_cases()[0]
        evaluator = cached_model_evaluator('boiling1d')

        losses: list[tuple[Fraction, float, str]] = []
        for fraction in FRACTIONS:
            preprocessors = default_boiling_preprocessors(
                direct_visualization=direct,
                visualization_window_width=fraction,
            )

            datasets = get_image_dataset(
                case(),
                transformers=preprocessors,
                experiment='boiling1d',
                cache_stages=(0,),  # do not cache visualization window datasets
            )

            model = get_baseline_architecture(
                datasets, strategy=strategy, normalize_images=True
            ) | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )

            fit_model = fit_boiling_model(
                model,
                datasets,
                get_baseline_fit_params(),
                target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
                strategy=strategy,
            )

            compiled_model = fit_model.architecture | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )
            evaluation = evaluator(compiled_model, datasets)

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

        plot_data = pd.DataFrame(
            losses, columns=['visualization window fraction', 'loss', 'subset']
        )
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        sns.scatterplot(
            ax=ax, data=plot_data, x='visualization window fraction', y='loss', hue='subset'
        )
        ax.set_xlabel('Visualization window fraction')
        ax.set_ylabel('Loss')
        ax.set_xscale('log')
        ax.set_yscale('log')

        figure_path = resolve(
            _learning_curve_study_path() / f'boiling1d-{direct_label}.png',
            parents=True,
        )
        f.savefig(str(figure_path))


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _learning_curve_study_path() -> Path:
    return studies_path() / 'visualization-window'
