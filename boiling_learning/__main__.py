import typer

from boiling_learning.app.examples import image_standardization, uncertainties
from boiling_learning.app.figures import (
    architectures,
    boiling_curve,
    diameter_effects,
    inclination_effects,
)
from boiling_learning.app.studies import (
    animate,
    automl,
    automl_cross_surface,
    automl_learning_curve,
    automl_strategies,
    automl_transfer_learning_curve,
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
    single_surface,
    transfer_learning_curve,
    validate,
    visualization_window,
    visualization_window_multi_surface,
)

app = typer.Typer()
app.add_typer(animate.app, name='animate')
app.add_typer(architectures.app, name='architectures')
app.add_typer(automl.app, name='automl')
app.add_typer(automl_cross_surface.app, name='automl-cross-surface')
app.add_typer(automl_learning_curve.app, name='automl-learning-curve')
app.add_typer(automl_strategies.app, name='automl-strategies')
app.add_typer(automl_transfer_learning_curve.app, name='automl-transfer-learning-curve')
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
app.add_typer(image_standardization.app, name='image-standardization')
app.add_typer(inclination_effects.app, name='inclination-effects')
app.add_typer(learning_curve.app, name='learning-curve')
app.add_typer(preprocessing.app, name='preprocessing')
app.add_typer(single_surface.app, name='single-surface')
app.add_typer(transfer_learning_curve.app, name='transfer-learning-curve')
app.add_typer(uncertainties.app, name='uncertainties')
app.add_typer(validate.app, name='validate')
app.add_typer(visualization_window.app, name='visualization-window')
app.add_typer(visualization_window_multi_surface.app, name='visualization-window-multi-surface')


if __name__ == '__main__':
    app()
