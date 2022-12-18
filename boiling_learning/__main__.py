import typer

from boiling_learning.app.studies import (
    animate,
    automl,
    cross_surface,
    downscaling_preprocessing,
    downscaling_training,
    example_frames,
    image_normalization,
    learning_curve,
    preprocessing,
    transfer_learning_curve,
    validate,
)

app = typer.Typer()
app.add_typer(animate.app, name='animate')
app.add_typer(automl.app, name='automl')
app.add_typer(cross_surface.app, name='cross-surface')
app.add_typer(downscaling_preprocessing.app, name='downscaling-preprocessing')
app.add_typer(downscaling_training.app, name='downscaling-training')
app.add_typer(example_frames.app, name='example-frames')
app.add_typer(image_normalization.app, name='image-normalization')
app.add_typer(learning_curve.app, name='learning-curve')
app.add_typer(preprocessing.app, name='preprocessing')
app.add_typer(transfer_learning_curve.app, name='transfer-learning-curve')
app.add_typer(validate.app, name='validate')


if __name__ == '__main__':
    app()
