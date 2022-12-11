import rich
import typer

from boiling_learning.app.automl.autofit_dataset import autofit_dataset
from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    get_baseline_boiling_architecture,
)

app = typer.Typer()
console = rich.console.Console()


@app.command()
def boiling1d(
    direct: bool = typer.Option(..., '--direct/--indirect'),
    normalize: bool = typer.Option(...),
) -> None:
    """Validate current implementation against reference."""
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

    model = autofit_dataset(
        baseline_boiling_dataset(direct_visualization=direct),
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        normalize_images=normalize,
        max_model_size=baseline_architecture_size,
        goal=None,
        experiment='boiling1d',
        strategy=strategy,
    )

    console.print(model.validation_metrics)
    console.print(model.test_metrics)


@app.command()
def condensation() -> None:
    pass
