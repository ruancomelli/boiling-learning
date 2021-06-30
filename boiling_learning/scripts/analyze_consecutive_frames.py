import math
from functools import partial
from typing import Callable, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from boiling_learning.preprocessing.image import ensure_grayscale


def main(
    frames: Iterable[np.ndarray],
    timeshifts: Iterable[int],
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    final_timeshift: int = 1,
    xscale: str = 'linear',
) -> None:
    frames = list(map(np.squeeze, map(ensure_grayscale, frames)))

    ref: np.ndarray = frames[0]
    timeshifts: List[int] = sorted(
        frozenset(timeshifts) | {0, final_timeshift}
    )
    timeshifted: List[np.ndarray] = list(map(frames.__getitem__, timeshifts))

    for name, scorer in metrics.items():
        evaluations = list(map(partial(scorer, ref), timeshifted))

        evaluations_dict = dict(zip(timeshifts, evaluations))
        original_evaluation = evaluations_dict[0]
        final_evaluation = evaluations_dict[final_timeshift]

        fig, ax = plt.subplots()
        ax.plot(timeshifts, evaluations, 'k.')
        ax.plot([0], [original_evaluation], 'r.')
        ax.axhline(original_evaluation, linestyle='--', color='gray')

        ax.set_xlabel('Frame #')

        ax.set_title(
            f'{name} (Frame #{final_timeshift} = {final_evaluation / original_evaluation:.0%})'
        )
        ax.set_xscale(xscale)

        _, top = ax.get_ylim()
        ax.set_ylim(0, math.ceil(top))

        fig.show()

    # ------------------------------------------
    # PART 2
    # ------------------------------------------
    fig = plt.figure()

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(ref, cmap='gray')
    ax.set_title('Frame #0')

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(frames[final_timeshift], cmap='gray')
    ax.set_title(f'Frame #{final_timeshift}')


if __name__ == '__main__':
    raise RuntimeError(
        'Cannot execute consecutive frames analysis without images!'
    )
