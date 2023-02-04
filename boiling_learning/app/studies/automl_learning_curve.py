from fractions import Fraction

import typer
from loguru import logger
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.automl.autofit_dataset import autofit_dataset
from boiling_learning.app.automl.evaluation import cached_best_model_evaluator
from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    fit_boiling_model,
    get_pretrained_baseline_boiling_model,
)
from boiling_learning.app.training.common import (
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.app.training.evaluation import cached_model_evaluator
from boiling_learning.model.training import compile_model
from boiling_learning.transforms import dataset_sampler

app = typer.Typer()
console = Console()
FRACTIONS = (Fraction(1, 100), 1)


@app.command()
@logger.catch
def boiling1d(model_size_reduce: int = 1) -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )
    model_evaluator = cached_model_evaluator('boiling1d')
    best_model_evaluator = cached_best_model_evaluator(
        experiment='boiling1d',
        strategy=strategy,
    )

    tables: list[Table] = []
    for direct in False, True:
        direct_label = 'direct' if direct else 'indirect'

        datasets = baseline_boiling_dataset(direct_visualization=direct)

        for fraction in FRACTIONS:
            subsampled = (
                datasets | dataset_sampler(count=fraction, subset='train')
                if fraction != 1
                else datasets
            )

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

            baseline_architecture_size = baseline_model_evaluation.trainable_parameters_count

            hypermodel = autofit_dataset(
                subsampled,
                target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
                normalize_images=True,
                max_model_size=baseline_architecture_size
                if model_size_reduce == 1
                else baseline_architecture_size // model_size_reduce,
                goal=None,
                experiment='boiling1d',
                strategy=strategy,
            )

            best_model = best_model_evaluator(
                hypermodel,
                datasets,
                measure_uncertainty=False,
            )

            fit_model = fit_boiling_model(
                best_model
                | compile_model(
                    **get_baseline_compile_params(strategy=strategy),
                ),
                datasets,
                get_baseline_fit_params(),
                target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
                strategy=strategy,
            )

            compiled_model = fit_model.architecture | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )
            best_model_evaluation = model_evaluator(
                compiled_model,
                datasets,
                measure_uncertainty=False,
            )

            best_model_table = Table(
                'Metric',
                'Training',
                'Validation',
                'Test',
                title=f'AutoML best model metrics - {direct_label} - {fraction}',
            )

            best_model_table.add_row(
                'Size',
                None,
                str(best_model_evaluation.trainable_parameters_count),
                None,
            )

            for (
                metric,
                training_metric,
                validation_metric,
                test_metric,
            ) in best_model_evaluation.iter_metrics():
                if metric == 'loss':
                    continue

                best_model_table.add_row(
                    metric,
                    str(training_metric),
                    str(validation_metric),
                    str(test_metric),
                )

            tables.append(best_model_table)

    console.print(Columns(tables))


@app.command()
def condensation() -> None:
    raise NotImplementedError
