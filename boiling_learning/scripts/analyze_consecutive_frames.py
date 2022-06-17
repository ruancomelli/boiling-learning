# import math
# from typing import Callable, Dict, Tuple

# import matplotlib.pyplot as plt
# import numpy as np

# from boiling_learning.preprocessing.image import grayscale


# def main(
#     frames: Dict[int, np.ndarray],
#     metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
#     final_timeshift: int = 1,
#     xscale: str = 'linear',
#     figsize: Tuple[int, int] = (7, 5),
# ) -> None:
#     if not {0, final_timeshift}.issubset(frames):
#         raise ValueError(
#             f'frames dictionary must contain the keys `0` and `{final_timeshift}`. '
#             f'Got {tuple(frames)}'
#         )

#     frames = {index: np.squeeze(grayscale(frame)) for index, frame in frames.items()}

#     # ------------------------------------------
#     # PART 1
#     # ------------------------------------------
#     for name, scorer in metrics.items():
#         evaluations = {index: scorer(frames[0], frame) for index, frame in frames.items()}

#         original_evaluation = evaluations[0]
#         final_evaluation = evaluations[final_timeshift]

#         fig, ax = plt.subplots(figsize=figsize)
#         ax.scatter(evaluations.keys(), evaluations.values(), color='k', s=15)
#         ax.scatter(0, original_evaluation, color='r', label='reference', s=25)
#         ax.scatter(final_timeshift, final_evaluation, color='b', label='target', s=25)
#         ax.legend()

#         ax.set_xlabel('Frame #')

#         ax.set_title(
#             f'{name} (Frame #{final_timeshift} = {final_evaluation / original_evaluation:.0%})'
#         )
#         ax.set_xscale(xscale)

#         # _, right = ax.get_xlim()
#         _, top = ax.get_ylim()
#         # ax.set_xlim(0, right)
#         ax.set_ylim(0, math.ceil(top))
#         ax.grid(which='both', axis='both', alpha=0.5)

#         fig.show()

#     # ------------------------------------------
#     # PART 2
#     # ------------------------------------------
#     fig = plt.figure()

#     ax = fig.add_subplot(1, 2, 1)
#     ax.imshow(frames[0], cmap='gray')
#     ax.set_title('Frame #0')

#     ax = fig.add_subplot(1, 2, 2)
#     ax.imshow(frames[final_timeshift], cmap='gray')
#     ax.set_title(f'Frame #{final_timeshift}')


# if __name__ == '__main__':
#     raise RuntimeError('Cannot execute consecutive frames analysis without images!')
