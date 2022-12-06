import typer

from boiling_learning.app.studies import animate, downscaling, example_frames, validate

app = typer.Typer()
app.add_typer(animate.app, name='animate')
app.add_typer(downscaling.app, name='downscaling')
app.add_typer(example_frames.app, name='example-frames')
app.add_typer(validate.app, name='validate')


if __name__ == '__main__':
    app()
