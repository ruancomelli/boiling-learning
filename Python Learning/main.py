# from pathlib import Path
# import modin.pandas as pd

# key = 'Voltage [V]'
# df = pd.read_csv(Path('.') / 'experiments' / 'Experiment Output 2020-02-14' / 'Experiment 10-25 (0).csv', usecols=[key])
# print(df[key].max())

# if __name__ == '__main__':
#     import sandbox.mirror
#     import sandbox.filters
#     import sandbox.sandboxing

import os
from pathlib import Path

def add_to_system_path(path_to_add, add_if_exists=False):
    str_to_add = str(path_to_add)
    if add_if_exists or (str_to_add not in os.environ['PATH']):
        os.environ['PATH'] += os.pathsep + str_to_add

python_project_home_path = Path().absolute().resolve()
project_home_path = python_project_home_path.parent.resolve()

# ensure that anaconda is in system's PATH
# if os.environ['COMPUTERNAME'] == 'LABSOLAR29-001':
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3')
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'mingw-w64' / 'bin')
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'usr' / 'bin')
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'bin')
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Scripts')
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'bin')
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'condabin')

import sys

if __name__ == '__main__':
    # import run_experiment
    # import process_data
    print(sys.version_info)