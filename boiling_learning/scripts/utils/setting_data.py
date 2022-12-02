from boiling_learning.preprocessing.experiment_video import ExperimentVideo


def check_experiment_video_dataframe_indices(ev: ExperimentVideo) -> None:
    indices = tuple(map(int, ev.df[ev.column_names.index]))
    expected = tuple(range(len(ev.frames())))
    if indices != expected:
        raise ValueError(
            f'expected indices != indices for {ev.name}.'
            f' Got expected: {expected}.'
            f' Got indices: {indices}.'
        )
