import rich
import typer

from boiling_learning.app.configuration import configure
from boiling_learning.app.training.boiling1d import get_pretrained_baseline_boiling_model
from boiling_learning.app.training.condensation import get_pretrained_baseline_condensation_model

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

    console.print(model.validation_metrics)
    console.print(model.test_metrics)


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
