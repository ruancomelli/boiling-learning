# import rich
# import typer

# from boiling_learning.app.training.boiling import get_pretrained_baseline_boiling_model

# app = typer.Typer()
# console = rich.console.Console()


# def validate(
#     direct: bool = typer.Option(..., '--direct/--indirect'),
#     normalize: bool = typer.Option(...),
# ) -> None:
#     """Validate current implementation against reference."""
#     model = get_pretrained_baseline_boiling_model(direct=direct, normalize_images=normalize)
#     console.print(model.evaluation)
