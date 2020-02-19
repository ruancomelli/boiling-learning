import os
import pathlib

# TODO: use collections.ChainMap

# if os.environ['COMPUTERNAME'] == 'LABSOLAR29-001':
#     os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3')
#     os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'mingw-w64' / 'bin')
#     os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'usr' / 'bin')
#     os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'bin')
#     os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Scripts')
#     os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'bin')
#     os.environ["PATH"] += os.pathsep + str(pathlib.Path('C:') / 'Users' / 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'condabin')

import numpy as np

from functools import partial
import tensorflow as tf

from tensorflow.python.keras.metrics import (
    accuracy, # fbeta_score, 
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import utils
import management
from management import Mirror

from model_definitions import KramerNet, LinearRegression

import pprint
pp = pprint.PrettyPrinter(indent=4, width=160)

save_map = {
    'model': management.save_keras_model,
    'history': utils.fold
    # 'history': management.save_pkl
}
load_map = {
    'model': management.load_keras_model,
    'history': utils.fold
    # 'history': management.load_pkl
}

manager = management.ModelManager(
    file_name_fmt='{index}.model',
    save_method=management.save_serialized(save_map),
    load_method=management.load_serialized(load_map),
    printer=pp.pprint,
    verbose=1
)

def gen_data(n_samples, x_scale=[0,1], noise=0.5):
    '''Generate univariate regression dataset'''
    x = np.sort(np.random.rand(n_samples))
    y = x + noise*np.random.randn(n_samples)
    x = x_scale[0] + (x_scale[1]-x_scale[0])*x
    X = x.reshape(-1,1)
    return X, y

data_params = dict(
    n_samples=1000,
    noise=0.1,
    x_scale=[0, 1]
)
X, y = gen_data(**data_params)

from sklearn.model_selection import train_test_split
X, X_val, y, y_val = train_test_split(X, y, train_size=0.8)

import matplotlib.pyplot as plt

optimizer_params = dict(
    lr=1e-2
)
reduce_lr_on_plateau_params = dict(
    monitor='val_loss', 
    factor=0.5, patience=5, min_delta=0.0001, min_lr=0, 
    verbose=1, mode='auto', cooldown=0
)
early_stopping_params = dict(
    monitor='val_loss', 
    min_delta=0, patience=20, baseline=None,
    verbose=1, mode='auto',
    restore_best_weights=True
)

linear_model_params = Mirror({'propagate': True},
    input_shape=1,
    compile_setup=Mirror({'propagate': True},
        do=True,
        params=Mirror({'propagate': True},
            optimizer=Mirror.Split(
                desc={
                    'name': 'adam',
                    'params': optimizer_params
                }, 
                value=Adam(**optimizer_params)
            ),
            loss='mse',
            # metrics=['val_loss']
        )
    ),
    fit_setup=Mirror({'propagate': True},
        do=True,
        method='fit',
        params=Mirror({'propagate': True},
            x=Mirror.Split(
                desc=data_params,
                value=X
            ),
            y=Mirror.Split(
                desc=data_params,
                value=y
            ),
            epochs=100, 
            verbose=2, 
            callbacks=Mirror.Split(
                desc=[
                    {
                        'name': 'ModelCheckpoint',
                        'params': {
                            'save_best_only': True
                        }
                    },
                    {
                        'name': 'ModelCheckpoint',
                        'params': {
                            'save_best_only': False
                        }
                    },
                    {
                        'name': 'EarlyStopping',
                        'params': early_stopping_params
                    },
                    {
                        'name': 'ReduceLROnPlateau',
                        'params': reduce_lr_on_plateau_params
                    }
                ],
                value=[
                    ModelCheckpoint(
                        str(manager.models_path / 'BestValidation.h5'),
                        monitor='val_loss', verbose=1, 
                        save_best_only=True
                    ),
                    ModelCheckpoint(
                        str(manager.models_path / 'LastTrained.h5'),
                        monitor='val_loss', verbose=0,
                        save_best_only=False
                    ),
                    EarlyStopping(**early_stopping_params),
                    ReduceLROnPlateau(**reduce_lr_on_plateau_params)
                ]
            ),
            validation_data=Mirror.Split(
                desc=['X_val', 'y_val'],
                value=(X_val, y_val)
            ),
            shuffle=True
        )
    )
)

linear_model = manager.provide_model(
    LinearRegression.creator,
    params=linear_model_params,
    save=True,
    load=True
)['model']

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linear_model_sk = LinearRegression().fit(X, y)

y_pred = linear_model.predict(X)
y_pred_sk = linear_model_sk.predict(X)

mse = mean_squared_error(y, y_pred)
mse_sk = mean_squared_error(y, y_pred_sk)

print('Linear model MSE:',mse)
print('1D Fit MSE:', mse_sk)
print('Relative diff:', 100*(mse - mse_sk)/mse_sk, '%')

plt.plot(X, y, '.', label='Data')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.plot(X, y_pred, '.', label='Linear model')
plt.plot(X, y_pred_sk, '.', label='1D fit')
plt.legend()
plt.show()

y_pred_val = linear_model.predict(X_val)
y_pred_sk_val = linear_model_sk.predict(X_val)

mse = mean_squared_error(y_val, y_pred_val)
mse_sk = mean_squared_error(y_val, y_pred_sk_val)

print('Linear model MSE:',mse)
print('1D Fit MSE:', mse_sk)
print('Relative diff:', 100*(mse - mse_sk)/mse_sk, '%')

plt.plot(X_val, y_val, '.', label='Data')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.plot(X_val, y_pred_val, '.', label='Linear model')
plt.plot(X_val, y_pred_sk_val, '.', label='1D fit')
plt.legend()
plt.show()

