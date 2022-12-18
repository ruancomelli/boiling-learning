from fractions import Fraction

import rich
import typer

from boiling_learning.app.automl.autofit_dataset import autofit_dataset
from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import DEFAULT_CONDENSATION_MASS_RATE_TARGET
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.datasets.preprocessed.condensation import condensation_dataset
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    get_baseline_boiling_architecture,
)
from boiling_learning.app.training.condensation import get_baseline_condensation_architecture
from boiling_learning.transforms import dataset_sampler

app = typer.Typer()
console = rich.console.Console()


@app.command()
def boiling1d(
    direct: bool = typer.Option(..., '--direct/--indirect'),
    normalize: bool = typer.Option(...),
    each: int = typer.Option(1),
) -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    baseline_architecture = get_baseline_boiling_architecture(
        direct_visualization=direct,
        normalize_images=normalize,
        strategy=strategy,
    )

    baseline_architecture_size = int(
        baseline_architecture.count_parameters(
            trainable=True,
            non_trainable=False,
        )
    )

    datasets = baseline_boiling_dataset(direct_visualization=direct)

    if each != 1:
        datasets = datasets | dataset_sampler(Fraction(1, each), subset='train')

    autofit_result = autofit_dataset(
        datasets,
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        normalize_images=normalize,
        max_model_size=baseline_architecture_size,
        goal=None,
        experiment='boiling1d',
        strategy=strategy,
    )

    console.print(autofit_result.tune_model_return.validation_metrics)
    console.print(autofit_result.tune_model_return.test_metrics)


@app.command()
def condensation(normalize: bool = typer.Option(...)) -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    baseline_architecture = get_baseline_condensation_architecture(
        normalize_images=normalize,
        strategy=strategy,
    )

    baseline_architecture_size = int(
        baseline_architecture.count_parameters(
            trainable=True,
            non_trainable=False,
        )
    )

    ds_train, ds_val, ds_test = condensation_dataset(each=30)()

    print(len(ds_train))
    print(len(ds_val))
    print(len(ds_test))

    assert False

    autofit_result = autofit_dataset(
        condensation_dataset(each=30),
        target=DEFAULT_CONDENSATION_MASS_RATE_TARGET,
        normalize_images=normalize,
        max_model_size=baseline_architecture_size,
        goal=None,
        experiment='condensation',
        strategy=strategy,
    )

    console.print(autofit_result.tune_model_return.validation_metrics)
    console.print(autofit_result.tune_model_return.test_metrics)
