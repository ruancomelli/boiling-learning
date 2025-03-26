from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import figures_path
from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.displaying import glossary, units
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
from boiling_learning.model.evaluate import UncertainValue
from boiling_learning.model.training import compile_model
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()
console = Console()

OUTLIER_LOSS = 500


@app.command()
def boiling1d(
    factors: list[int] = typer.Option(list(range(1, 7))),
) -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    case = boiling_cases()[0]
    evaluator = cached_model_evaluator("boiling1d")
    tables: list[Table] = []

    for direct in False, True:
        direct_label = "direct" if direct else "indirect"

        table = Table(
            "Factor",
            "Training",
            "Validation",
            "Test",
            title=f"Downscaling analysis - {direct_label}",
        )

        evaluations: list[tuple[int, str, UncertainValue]] = []

        for factor in factors:
            preprocessors = default_boiling_preprocessors(
                direct_visualization=direct,
                downscale_factor=factor,
            )
            datasets = get_image_dataset(
                case(),
                transformers=preprocessors,
                experiment="boiling1d",
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
                try_id=1 if factor == 2 else 0,  # noqa: PLR2004
            )

            compiled_model = fit_model.architecture | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )
            evaluation = evaluator(compiled_model, datasets)

            evaluations.extend(
                (
                    (factor, "Training", evaluation.training_metrics["MSE"].value),
                    (factor, "Validation", evaluation.validation_metrics["MSE"].value),
                    (factor, "Test", evaluation.test_metrics["MSE"].value),
                )
            )

            table.add_row(
                f"{factor}",
                f"{evaluation.training_metrics['MSE']}",
                f"{evaluation.validation_metrics['MSE']}",
                f"{evaluation.test_metrics['MSE']}",
            )

        tables.append(table)

        plot_data = pd.DataFrame(
            evaluations, columns=["Downscaling factor", "Subset", "Loss"]
        )
        f, ax = plt.subplots(1, 1, figsize=(3, 3))

        sns.scatterplot(
            ax=ax,
            data=plot_data[plot_data["Loss"] < OUTLIER_LOSS],
            x="Downscaling factor",
            y="Loss",
            hue="Subset",
            alpha=0.75,
        )
        outliers = plot_data[plot_data["Loss"] >= OUTLIER_LOSS]["Downscaling factor"]
        for outlier in outliers:
            ax.axvspan(outlier - 0.1, outlier + 0.1, color="red", alpha=0.15, hatch="/")

        ax.grid(which="minor", axis="x")
        ax.set(
            xlabel=f"Downscaling factor, ${glossary['downscaling factor']}$",
            ylabel=f"MSE [{units['mse']}]",
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(ScalarFormatter())

        ax.yaxis.set_minor_formatter(
            lambda val, pos: int(val) if abs(val - int(val)) < 1e-6 else val  # noqa: PLR2004
        )
        ax.yaxis.set_major_formatter(
            lambda val, pos: int(val) if abs(val - int(val)) < 1e-6 else val  # noqa: PLR2004
        )

        save_figure(
            f, _downscaling_training_study_path() / f"boiling1d-{direct_label}.pdf"
        )
        save_figure(
            f,
            _downscaling_training_study_figures_path()
            / f"boiling1d-{direct_label}.pdf",
        )

    console.print(Columns(tables))


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _downscaling_training_study_figures_path() -> Path:
    return resolve(figures_path() / "results" / "downscaling", dir=True)


def _downscaling_training_study_path() -> Path:
    return resolve(studies_path() / "downscaling-training", dir=True)
