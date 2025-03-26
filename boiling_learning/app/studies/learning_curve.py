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
from boiling_learning.app.datasets.preprocessed.boiling1d import (
    baseline_boiling_dataset,
)
from boiling_learning.app.displaying import units
from boiling_learning.app.displaying.figures import save_figure
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

FRACTIONS = tuple(Fraction(i, 100) for i in range(1, 10)) + tuple(
    Fraction(i, 10) for i in range(1, 11)
)


@app.command()
def boiling1d() -> None:
    logger.info("Analyzing learning curve")

    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    evaluator = cached_model_evaluator("boiling1d")

    tables: list[Table] = []
    for direct in False, True:
        direct_label = "direct" if direct else "indirect"

        datasets = baseline_boiling_dataset(direct_visualization=direct)

        table = Table(
            "Subsample",
            "Training loss",
            "Validation loss",
            "Test loss",
            title=f"Learning curve - {direct_label}",
        )

        losses: list[tuple[Fraction, float, str]] = []
        for fraction in FRACTIONS:
            subsampled = (
                datasets | dataset_sampler(count=fraction, subset="train")
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
                try_id=4,
            )

            compiled_model = fit_model.architecture | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )
            evaluation = evaluator(
                compiled_model,
                subsampled,
                measure_uncertainty=False,
            )

            table.add_row(
                f"{fraction} ({float(fraction):.0%})",
                f"{evaluation.training_metrics['MSE']}",
                f"{evaluation.validation_metrics['MSE']}",
                f"{evaluation.test_metrics['MSE']}",
            )

            losses.extend(
                (
                    (fraction, evaluation.training_metrics["MSE"], "Training"),
                    (fraction, evaluation.validation_metrics["MSE"], "Validation"),
                    (fraction, evaluation.test_metrics["MSE"], "Test"),
                )
            )

        tables.append(table)

        plot_data = pd.DataFrame(losses, columns=["fraction", "loss", "Subset"])
        f, ax = plt.subplots(1, 1, figsize=(2.8, 2.8))
        sns.scatterplot(
            ax=ax,
            data=plot_data,
            x="fraction",
            y="loss",
            hue="Subset",
            alpha=0.75,
        )
        ax.set(
            xlabel="Dataset subsample size",
            ylabel=f"MSE [{units['mse']}]",
            xscale="log",
            # yscale='log',
        )
        # ax.xaxis.set_minor_formatter(PercentFormatter(xmax=max(map(float, FRACTIONS))))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=max(map(float, FRACTIONS))))
        # ax.yaxis.set_minor_formatter(lambda value, pos: int(value))
        ax.yaxis.set_major_formatter(lambda value, pos: int(value))

        save_figure(
            f,
            _learning_curve_study_path() / f"boiling1d-{direct_label}.pdf",
        )
        save_figure(
            f,
            _learning_curve_figures_path() / f"boiling1d-{direct_label}.pdf",
        )
        save_figure(
            f,
            _learning_curve_study_path() / f"boiling1d-{direct_label}.png",
        )

    console.print(Columns(tables))


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _learning_curve_study_path() -> Path:
    return resolve(studies_path() / "learning-curve", dir=True)


def _learning_curve_figures_path() -> Path:
    return resolve(figures_path() / "results" / "learning-curve", dir=True)
