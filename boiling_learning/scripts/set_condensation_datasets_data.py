import re
from datetime import timedelta
from typing import Dict, Iterable, Optional

import parse
import yaml

from boiling_learning.preprocessing.ExperimentVideo import ExperimentVideo
from boiling_learning.preprocessing.ImageDataset import ImageDataset
from boiling_learning.preprocessing.video import get_fps
from boiling_learning.utils.utils import (
    PathLike,
    ensure_resolved,
    print_header,
    print_verbose,
)

_subcase_patterns = {
    'T_inf': parse.compile('T_inf {:d}C'),
    'T_s': parse.compile('T_s {:d}C'),
    'rh': parse.compile('rh {:d}%'),
}
_timedelta_pattern = re.compile(r'(?P<h>\d{2}):(?P<min>\d{2}):(?P<s>\d{2})')


def _parse_timedelta(s: Optional[str]) -> Optional[timedelta]:
    if s is None:
        return None

    m = _timedelta_pattern.fullmatch(s)
    return timedelta(
        hours=int(m['h']), minutes=int(m['min']), seconds=int(m['s'])
    )


def main(
    datasets: Iterable[ImageDataset],
    dataspecpath: PathLike,
    verbose: bool = False,
) -> Dict[str, ImageDataset]:
    dataspecpath = ensure_resolved(dataspecpath)
    dataspec = yaml.safe_load(dataspecpath.read_text())

    datasets_dict = {}
    for dataset in datasets:
        for ev_name, ev in dataset.items():
            case, subcase, test_name, video_name = ev_name.split(':')

            # TODO: add average mass rate to categories?

            if case == 'stainless steel' and subcase == 'polished':
                # those are the standard conditions for polished SS
                # temperatures are in Celsius
                # relative humidity is in percentage
                # age can be either "new" or "old"
                categories = {'age': 'new', 'T_inf': 50, 'T_s': 10, 'rh': 80}
            elif case == 'parametric':
                if subcase == 'old':
                    categories = {
                        'age': 'old',
                        'T_inf': 50,
                        'T_s': 10,
                        'rh': 80,
                    }
                else:
                    categories = {
                        'age': 'new',
                        'T_inf': 50,
                        'T_s': 10,
                        'rh': 80,
                    }
                    for change, pattern in _subcase_patterns.items():
                        match = pattern.search(subcase)
                        if match is not None:
                            categories[change] = match[0]
                            break
            else:
                continue

            videospec = dataspec['cases'][case]['subcases'][subcase]['tests'][
                test_name
            ]['videos'][video_name + '.mp4']

            videodata = ExperimentVideo.VideoData(
                categories=categories,
                fps=get_fps(ev.video_path),
                # since there is no syncing between video and experimental data here,
                # we simply set the first frame as the reference
                ref_index=0,
                ref_elapsed_time=0,
                start_elapsed_time=_parse_timedelta(videospec['start']),
                end_elapsed_time=_parse_timedelta(videospec['end']),
            )
            ev.set_video_data(videodata)
            print_verbose(
                verbose, f'{ev_name} -> [{ev.start}, {ev.end}) :: {categories}'
            )

            dataset_name = ':'.join((case, subcase))
            datasets_dict.setdefault(
                dataset_name, ImageDataset(dataset_name)
            ).add(ev)

    for dataset in datasets_dict.values():
        if verbose:
            print_header(dataset.name)

        try:
            print_verbose(verbose, 'Loading')
            dataset.load_dfs(overwrite=False, missing_ok=False)
        except FileNotFoundError:
            print_verbose(verbose, 'Failed, making dataframes.')

            dataset.make_dataframe(
                recalculate=False,
                exist_load=True,
                enforce_time=True,
                categories_as_int=True,
                inplace=True,
            )

            for ev in dataset.values():
                _indices = tuple(map(int, ev.df[ev.column_names.index]))
                _expected = tuple(range(len(ev.video)))
                if _indices != _expected:
                    raise ValueError(
                        f'expected indices != indices for {ev.name}.'
                        f' Got expected: {_expected}'
                        f' Got indices: {_indices}'
                    )

            dataset.save_dfs(overwrite=False)

    print_verbose(verbose, 'Condensation datasets dict:')
    for ds_name, ds in datasets_dict.items():
        print_verbose(verbose, ds_name, '::', ds)

    return datasets_dict


if __name__ == '__main__':
    raise RuntimeError(
        '*set_condensation_datasets_data* cannot be executed as a standalone script yet.'
    )
