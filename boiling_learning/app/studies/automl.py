import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from matplotlib.collections import PathCollection
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.automl.autofit_dataset import autofit_dataset
from boiling_learning.app.automl.evaluation import cached_best_model_evaluator
from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import figures_path
from boiling_learning.app.datasets.preprocessed.boiling1d import (
    baseline_boiling_dataset,
)
from boiling_learning.app.displaying import units
from boiling_learning.app.displaying.figures import save_figure
from boiling_learning.app.figures.architectures import diagrams_path, model_to_tikz
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    get_pretrained_baseline_boiling_model,
)
from boiling_learning.app.training.common import get_baseline_compile_params
from boiling_learning.app.training.evaluation import cached_model_evaluator
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.training import compile_model
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()
console = Console()


@app.command()
@logger.catch
def boiling1d(model_size_reduce: int = 1) -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )
    model_evaluator = cached_model_evaluator("boiling1d")
    best_model_evaluator = cached_best_model_evaluator(
        experiment="boiling1d",
        strategy=strategy,
    )

    tables: list[Table] = []
    for direct in False, True:
        direct_label = "direct" if direct else "indirect"

        datasets = baseline_boiling_dataset(direct_visualization=direct)

        baseline_fit_return = get_pretrained_baseline_boiling_model(
            direct_visualization=direct,
            normalize_images=True,
            strategy=strategy,
        )
        baseline_model_evaluation = model_evaluator(
            baseline_fit_return.architecture
            | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            ),
            datasets,
            measure_uncertainty=False,
        )

        baseline_loss = baseline_model_evaluation.validation_metrics["MSE"]
        baseline_architecture_size = (
            baseline_model_evaluation.trainable_parameters_count
        )

        hypermodel = autofit_dataset(
            datasets,
            target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
            normalize_images=True,
            max_model_size=baseline_architecture_size
            if model_size_reduce == 1
            else baseline_architecture_size // model_size_reduce,
            goal=None,
            experiment="boiling1d",
            strategy=strategy,
        )

        best_model = best_model_evaluator(
            hypermodel,
            datasets,
            measure_uncertainty=False,
        )

        compiled_model = best_model | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )
        best_model_evaluation = model_evaluator(
            compiled_model,
            datasets,
            measure_uncertainty=False,
        )

        best_model_size_table = Table(
            "Size",
            "Value",
            title=f"AutoML best model sizes - {direct_label} visualization",
        )
        best_model_size_table.add_row(
            "Trainable", str(best_model_evaluation.trainable_parameters_count)
        )
        best_model_size_table.add_row(
            "Total", str(best_model_evaluation.total_parameters_count)
        )
        best_model_size_table.add_row(
            "Relative trainable",
            f"{best_model_evaluation.trainable_parameters_count}/{baseline_architecture_size}",
        )
        tables.append(best_model_size_table)

        best_model_table = Table(
            "Metric",
            "Training",
            "Validation",
            "Test",
            title=f"AutoML best model metrics - {direct_label} visualization",
        )
        for metric in best_model_evaluation.metrics_names:
            if metric == "loss":
                continue

            best_model_table.add_row(
                metric,
                str(best_model_evaluation.training_metrics[metric]),
                str(best_model_evaluation.validation_metrics[metric]),
                str(best_model_evaluation.test_metrics[metric]),
            )
        tables.append(best_model_table)

        best_hyperparameters_table = Table(
            "Hyperparameter",
            "Value",
            title=f"AutoML best model hyperparameters - {direct_label} visualization",
        )
        best_hyperparameters = hypermodel.best_hyperparameters()
        for hyperparameter, value in best_hyperparameters.values.items():
            best_hyperparameters_table.add_row(
                hyperparameter,
                str(value),
            )
        tables.append(best_hyperparameters_table)

        evaluations: list[tuple[int, float, str, str]] = []
        for model in itertools.islice(
            hypermodel.iter_best_models(),
            # skip best model
            1,
            None,
        ):
            compiled_model = LazyDescribed.from_describable(model) | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )

            evaluation = model_evaluator(
                compiled_model, datasets, measure_uncertainty=False
            )

            evaluations.extend(
                (
                    (
                        evaluation.trainable_parameters_count,
                        evaluation.validation_metrics["loss"],
                        "trainable",
                        "Searched models",
                    ),
                    (
                        evaluation.total_parameters_count,
                        evaluation.validation_metrics["loss"],
                        "total",
                        "Searched models",
                    ),
                )
            )

        evaluations.extend(
            (
                (
                    baseline_model_evaluation.trainable_parameters_count,
                    baseline_model_evaluation.validation_metrics["loss"],
                    "trainable",
                    "Baseline model",
                ),
                (
                    baseline_model_evaluation.total_parameters_count,
                    baseline_model_evaluation.validation_metrics["loss"],
                    "total",
                    "Baseline model",
                ),
                (
                    best_model_evaluation.trainable_parameters_count,
                    best_model_evaluation.validation_metrics["loss"],
                    "trainable",
                    "Best model",
                ),
                (
                    best_model_evaluation.total_parameters_count,
                    best_model_evaluation.validation_metrics["loss"],
                    "total",
                    "Best model",
                ),
            )
        )

        df = pd.DataFrame(
            evaluations, columns=["Model size", "Validation loss", "Type", "Model"]
        )

        for weights_group in "trainable", "total":
            f, ax = plt.subplots(1, 1, figsize=(2.8, 2.8))
            ret = sns.scatterplot(
                df[df["Type"] == weights_group],
                x="Model size",
                y="Validation loss",
                hue="Model",
                alpha=0.75,
                ax=ax,
            )
            colors = [
                c.get_facecolor()[0]
                for c in ret.get_children()
                if isinstance(c, PathCollection)
            ]
            baseline_color = colors[2]
            ax.axvline(
                baseline_architecture_size,
                color=baseline_color,
                alpha=0.5,
                linestyle="--",
            )
            ax.axhline(baseline_loss, color=baseline_color, alpha=0.5, linestyle="--")
            ax.set(
                xscale="log",
                yscale="log",
                ylabel=f"Validation loss [{units['mse']}]",
            )
            save_figure(
                f,
                _automl_study_path()
                / f"boiling1d-{direct_label}-{weights_group}-reduce{model_size_reduce}.pdf",
            )
            save_figure(
                f,
                _automl_figures_path()
                / f"boiling1d-{direct_label}-{weights_group}-reduce{model_size_reduce}.pdf",
            )

        (
            _automl_diagrams_path()
            / f"best-automl-{direct_label}-reduce{model_size_reduce}.tex"
        ).write_text(
            "\n".join(
                model_to_tikz(
                    best_model(),
                    max_rows_per_column=7,
                    standalone=False,
                )
            )
        )

    console.print(Columns(tables))


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _automl_study_path() -> Path:
    return resolve(studies_path() / "automl", dir=True)


def _automl_figures_path() -> Path:
    return resolve(figures_path() / "results" / "automl", dir=True)


def _automl_diagrams_path() -> Path:
    return resolve(diagrams_path() / "results" / "automl", dir=True)
