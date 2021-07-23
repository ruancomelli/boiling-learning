import math
from operator import itemgetter
from typing import Callable, Dict, FrozenSet, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from boiling_learning.preprocessing.image import ensure_grayscale


def main(
    frames: Iterable[Tuple[int, np.ndarray]],
    timeshifts: Iterable[int],
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    final_timeshift: int = 1,
    xscale: str = 'linear',
) -> None:
    timeshifts: FrozenSet[int] = frozenset(timeshifts) | {0, final_timeshift}

    frames = ((index, frame) for index, frame in frames if index in timeshifts)
    frames = (
        (index, np.squeeze(ensure_grayscale(frame))) for index, frame in frames
    )
    frames = sorted(frames, key=itemgetter(0))
    frames: Dict[int, np.ndarray] = dict(frames)

    ref: np.ndarray = frames[0]
    final: np.ndarray = frames[final_timeshift]

    for name, scorer in metrics.items():
        evaluations = {
            index: scorer(ref, frame) for index, frame in frames.items()
        }

        original_evaluation = evaluations[0]
        final_evaluation = evaluations[final_timeshift]

        fig, ax = plt.subplots()
        ax.axhline(original_evaluation, linestyle='--', color='gray')
        ax.scatter(evaluations.keys(), evaluations.values(), color='k', s=15)
        ax.scatter(0, original_evaluation, color='r', label='reference', s=25)
        ax.scatter(
            final_timeshift, final_evaluation, color='b', label='final', s=25
        )
        ax.legend()

        ax.set_xlabel('Frame #')

        ax.set_title(
            f'{name} (Frame #{final_timeshift} = {final_evaluation / original_evaluation:.0%})'
        )
        ax.set_xscale(xscale)

        _, right = ax.get_xlim()
        _, top = ax.get_ylim()
        ax.set_xlim(0, right)
        ax.set_ylim(0, math.ceil(top))

        fig.show()

    # ------------------------------------------
    # PART 2
    # ------------------------------------------
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(ref, cmap='gray')
    ax.set_title('Frame #0')

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(final, cmap='gray')
    ax.set_title(f'Frame #{final_timeshift}')


if __name__ == '__main__':
    raise RuntimeError(
        'Cannot execute consecutive frames analysis without images!'
    )
