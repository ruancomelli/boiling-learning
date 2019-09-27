import os
import pathlib

if os.environ['COMPUTERNAME'] == 'LABSOLAR29-001':
    os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3')
    os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'mingw-w64' / 'bin')
    os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'usr' / 'bin')
    os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'bin')
    os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Scripts')
    os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'bin')
    os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'condabin')

import keras

