from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from matplotlib.ticker import PercentFormatter
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import figures_path
from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.displaying import units
from boiling_learning.app.displaying.figures import save_figure
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

app = typer.Typer()
console = Console()

FRACTIONS = sorted(Fraction(num, 100) for num in range(100, 9, -10))
OUTLIER_LOSS = 1000


@app.command()
@logger.catch
def boiling1d() -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    tables: list[Table] = []

    evaluator = cached_model_evaluator('boiling1d')

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

        losses: list[tuple[Fraction, float, str]] = []
        for fraction in sorted(FRACTIONS):
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
                try_id=(2 if direct else 0),
            )

            compiled_model = fit_model.architecture | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )
            evaluation = evaluator(compiled_model, datasets, measure_uncertainty=False)

            table.add_row(
                f'{fraction} ({float(fraction):.0%})',
                f'{evaluation.training_metrics["MSE"]}',
                f'{evaluation.validation_metrics["MSE"]}',
                f'{evaluation.test_metrics["MSE"]}',
            )

            losses.extend(
                (
                    (fraction, evaluation.training_metrics['MSE'], 'Training'),
                    (fraction, evaluation.validation_metrics['MSE'], 'Validation'),
                    (fraction, evaluation.test_metrics['MSE'], 'Test'),
                )
            )

        tables.append(table)

        plot_data = pd.DataFrame(
            losses, columns=['Visualization window fraction', 'Loss', 'Subset']
        )

        f, ax = plt.subplots(1, 1, figsize=(2.8, 2.8))
        sns.scatterplot(
            ax=ax,
            data=plot_data[plot_data['Loss'] < OUTLIER_LOSS],
            x='Visualization window fraction',
            y='Loss',
            hue='Subset',
            alpha=0.75,
        )
        ax.set(ylabel=f'Loss [{units["mse"]}]')

        outliers = plot_data[plot_data['Loss'] >= OUTLIER_LOSS]['Visualization window fraction']
        for outlier in outliers:
            ax.axvspan(outlier - 0.015, outlier + 0.015, color='red', alpha=0.15, hatch='/')
        # ax.set(xscale='linear', yscale='log', xticks=FRACTIONS)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=max(map(float, FRACTIONS))))
        ax.yaxis.set_major_formatter(lambda value, pos: int(value))

        save_figure(f, _visualization_window_study_path() / f'boiling1d-{direct_label}.pdf')
        save_figure(f, _visualization_window_figures_path() / f'boiling1d-{direct_label}.pdf')

    console.print(Columns(tables))


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _visualization_window_figures_path() -> Path:
    return figures_path() / 'results' / 'visualization-window'


def _visualization_window_study_path() -> Path:
    return studies_path() / 'visualization-window'
