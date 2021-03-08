from parse import parse

import boiling_learning.utils


def chunkify(index, chunk_size):
    min_index = (index // chunk_size) * chunk_size
    max_index = min_index + chunk_size - 1
    return min_index, max_index

for subcase in case.frames_path.iterdir():
    boiling_learning.utils.print_header(subcase)
    for old_path in subcase.iterdir():
        parse_result = parse('{}_frame{index:d}.png', old_path.name)

        if parse_result:
            print(parse_result)
            index = parse_result['index']
            new_path = (
                old_path.parent
                / Path('from_{:0>5d}_to_{:0>5d}'.format(*chunkify(index, 100)))
                / old_path.name
            )
            print(f'{index}:')
            print(f'{old_path}')
            print(f'-> {new_path}')

            # new_path.parent.mkdir(parents=True, exist_ok=True)
            # old_path.rename(new_path)
