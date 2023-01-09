import typer

from boiling_learning.app.figures import boiling_curve, diameter_effects, inclination_effects
from boiling_learning.app.studies import (
    animate,
    automl,
    automl_strategies,
    consecutive_frames,
    cross_surface,
    data_augmentation,
    data_augmentation_automl,
    data_split,
    dataset_sizes,
    downscaling_preprocessing,
    downscaling_training,
    example_frames,
    example_frames_matrix,
    heat_flux_levels,
    image_brightness,
    image_normalization,
    learning_curve,
    preprocessing,
    transfer_learning_curve,
    validate,
    visualization_window,
)

app = typer.Typer()
app.add_typer(animate.app, name='animate')
app.add_typer(automl.app, name='automl')
app.add_typer(automl_strategies.app, name='automl-strategies')
app.add_typer(boiling_curve.app, name='boiling-curve')
app.add_typer(consecutive_frames.app, name='consecutive-frames')
app.add_typer(cross_surface.app, name='cross-surface')
app.add_typer(data_augmentation.app, name='data-augmentation')
app.add_typer(data_augmentation_automl.app, name='data-augmentation-automl')
app.add_typer(data_split.app, name='data-split')
app.add_typer(dataset_sizes.app, name='dataset-sizes')
app.add_typer(diameter_effects.app, name='diameter-effects')
app.add_typer(downscaling_preprocessing.app, name='downscaling-preprocessing')
app.add_typer(downscaling_training.app, name='downscaling-training')
app.add_typer(example_frames.app, name='example-frames')
app.add_typer(example_frames_matrix.app, name='example-frames-matrix')
app.add_typer(heat_flux_levels.app, name='heat-flux-levels')
app.add_typer(image_brightness.app, name='image-brightness')
app.add_typer(image_normalization.app, name='image-normalization')
app.add_typer(inclination_effects.app, name='inclination-effects')
app.add_typer(learning_curve.app, name='learning-curve')
app.add_typer(preprocessing.app, name='preprocessing')
app.add_typer(transfer_learning_curve.app, name='transfer-learning-curve')
app.add_typer(validate.app, name='validate')
app.add_typer(visualization_window.app, name='visualization-window')


if __name__ == '__main__':
    app()
