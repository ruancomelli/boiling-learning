import typer

from boiling_learning.app.studies import (
    animate,
    downscaling,
    example_frames,
    image_normalization,
    preprocessing,
    validate,
)

app = typer.Typer()
app.add_typer(animate.app, name='animate')
app.add_typer(downscaling.app, name='downscaling')
app.add_typer(example_frames.app, name='example-frames')
app.add_typer(image_normalization.app, name='image-normalization')
app.add_typer(preprocessing.app, name='preprocessing')
app.add_typer(validate.app, name='validate')


if __name__ == '__main__':
    app()
