import typer
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.automl.autofit_dataset import autofit_dataset
from boiling_learning.app.automl.evaluation import cached_best_model_evaluator
from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import (
    baseline_boiling_dataset,
)
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    get_pretrained_baseline_boiling_model,
)
from boiling_learning.app.training.common import get_baseline_compile_params
from boiling_learning.app.training.evaluation import cached_model_evaluator
from boiling_learning.automl.tuners import (
    EarlyStoppingBayesian,
    EarlyStoppingGreedy,
    EarlyStoppingHyperband,
)
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

    evaluator = cached_model_evaluator("boiling1d")
    best_model_evaluator = cached_best_model_evaluator(
        experiment="boiling1d",
        strategy=strategy,
    )

    tables: list[Table] = []
    for direct in False, True:
        direct_label = "direct" if direct else "indirect"

        for tuner_class in (
            EarlyStoppingGreedy,
            EarlyStoppingHyperband,
            EarlyStoppingBayesian,
        ):
            table = Table(
                "Validation loss",
                "Test loss",
                title=(
                    f"Automatic machine learning - {direct_label}"
                    f" visualization {tuner_class.__name__}"
                ),
            )

            datasets = baseline_boiling_dataset(direct_visualization=direct)

            baseline_fit_return = get_pretrained_baseline_boiling_model(
                direct_visualization=direct,
                normalize_images=True,
                strategy=strategy,
            )
            compiled_baseline_model = baseline_fit_return.architecture | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )
            baseline_evaluation = evaluator(
                compiled_baseline_model, datasets, measure_uncertainty=False
            )
            baseline_architecture_size = baseline_evaluation.trainable_parameters_count

            hypermodel = autofit_dataset(
                datasets,
                target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
                normalize_images=True,
                max_model_size=baseline_architecture_size,
                goal=None,
                experiment="boiling1d",
                strategy=strategy,
                tuner_class=tuner_class,
            )

            best_model = best_model_evaluator(
                hypermodel,
                datasets,
                measure_uncertainty=False,
            )

            compiled_best_model = best_model | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )
            best_model_evaluation = evaluator(
                compiled_best_model,
                datasets,
                measure_uncertainty=False,
            )

            table.add_row(
                str(best_model_evaluation.validation_metrics["MSE"]),
                str(best_model_evaluation.test_metrics["MSE"]),
            )
            tables.append(table)

    console.print(Columns(tables))


@app.command()
def condensation() -> None:
    pass
