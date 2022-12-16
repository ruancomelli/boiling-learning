import rich
import typer

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.bridging import to_tensorflow_triplet
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    DEFAULT_BOILING_OUTLIER_FILTER,
    get_pretrained_baseline_boiling_model,
)
from boiling_learning.app.training.common import get_baseline_compile_params
from boiling_learning.app.training.condensation import get_pretrained_baseline_condensation_model
from boiling_learning.model.evaluate import evaluate_with_uncertainty
from boiling_learning.model.training import compile_model

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

    model = get_pretrained_baseline_boiling_model(
        direct_visualization=direct,
        normalize_images=normalize,
        strategy=strategy,
    )

    _, ds_val, ds_test = to_tensorflow_triplet(
        baseline_boiling_dataset(direct_visualization=direct),
        prefilterer=DEFAULT_BOILING_OUTLIER_FILTER,
        batch_size=None,
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        experiment='boiling1d',
    )

    compiled_model = compile_model(
        model.architecture,
        get_baseline_compile_params(strategy=strategy),
    )

    validation_metrics = evaluate_with_uncertainty(compiled_model.architecture, ds_val())
    test_metrics = evaluate_with_uncertainty(compiled_model.architecture, ds_test())

    console.print(model.validation_metrics)
    console.print(model.test_metrics)

    console.print(validation_metrics)
    console.print(test_metrics)


@app.command()
def condensation(
    each: int = typer.Option(60),
    normalize: bool = typer.Option(...),
) -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    model = get_pretrained_baseline_condensation_model(
        each=each,
        normalize_images=normalize,
        strategy=strategy,
    )

    console.print(model.validation_metrics)
    console.print(model.test_metrics)
