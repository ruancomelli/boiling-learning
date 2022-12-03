import typer

from boiling_learning.app.studies.animate import animate
from boiling_learning.app.studies.display_example_frames import display_example_frames
from boiling_learning.app.studies.validation import validate

app = typer.Typer()
app.command()(animate)
app.command()(display_example_frames)
app.command()(validate)


if __name__ == '__main__':
    app()
