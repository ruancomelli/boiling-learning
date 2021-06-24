import more_itertools as mit

from boiling_learning.preprocessing import Case

case = Case()


def group_files_move(path, keyfunc, mover_gen):
    from boiling_learning.utils import group_files

    d = group_files(path, keyfunc)
    for key, lst in d.items():
        mov = mover_gen(key)
        for path in lst:
            new_path = mov(path)
            path.rename(new_path)


# pprint(
#     boiling_learning.utils.group_files(
#         project_path / 'testing_extract',
#         keyfunc=lambda x: chunkify(
#             parse('frame_{index:d}', x.name)['index'],
#             10
#         )
#     )
# )


for p in case.frames_path.iterdir():
    if mit.ilen(p.glob('*.png')) != 0:
        print(p)
        # group_files_move(
        #     p,
        #     keyfunc=lambda x: chunkify(
        #         parse('GOPR{:4d}_frame{index:d}.png', x.name)['index'],
        #         100
        #     ),
        #     mover_gen=lambda key: boiling_learning.utils.mover(
        #         p / f'from_{key[0]:0>5d}_to_{key[1]:0>5d}',
        #         make_dir=True
        #     )
        # )
