import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import tensorflow as tf

try:
    import keras
except ImportError:
    from tensorflow import keras

from pathlib import Path

import utils
from ModelManager import ModelCreator, ModelManager
from sklearn.base import BaseEstimator

manager = ModelManager(models_path=Path(__file__).parent / 'models', file_name_fmt='{index}.model')

class Model(BaseEstimator):
    # Linear regression with L2 regularization
    def __init__(self, lamb=0, solver='ne', lr=1, maxiter=1000, tol=1e-5, div_tol=1e6):
        # Initialization
        self.lamb = lamb
        self.solver = solver
        self.lr = lr
        self.maxiter = maxiter
        self.tol = tol
        self.div_tol = div_tol
        self.logger = {}
    
    def __add_ones(self, X):
        # Add column of ones
        X_new = np.c_[np.ones(X.shape[0]), X]
        return X_new
    
    def __fit_ne(self, X, y):
        """Fit the input data using the normal equations.
        """
        X = self.__add_ones(X)
        L = np.eye(X.shape[1])
        L[0][0] = 0
        try:
            self.w = np.linalg.solve(X.T @ X + self.lamb*L, X.T @ y)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('Model.__fit_ne: Singular matrix')
            else:
                raise

    def grad_J(self, X, y):
        """Compute the gradient of the cost function
        """
        m = y.size
        return 2/m * (
            X.T @ (X@self.w - y)
            + self.lamb*self.w
        )

    def hessian_J(self, X, y=None):
        m = y.size
        n = X.shape[1]
        return 2/m * (
            X.T @ X
            + self.lamb*np.eye(n)
        )

    def cond(self, X, y=None):
        return np.linalg.cond(
            self.hessian_J(X, y)
        )

    def __fit_gd(self, X, y):
        """Fit the input data using the gradient descent.
        """
        X_transformed = self.__add_ones(X)
        self.cond_hessian_J = self.cond(X_transformed, y)

        self.J_history = np.zeros(self.maxiter)
        self.w = np.zeros((X_transformed.shape[1],))
        
        for iter_count in range(self.maxiter):
            norm_w = np.linalg.norm(self.w)
            if norm_w > self.div_tol:
                iter_count -= 1
                self.J_history.resize(iter_count)
                self.logger['exit'] = 'divergence'
                self.logger['exit_message'] = f'Divergence found with np.linalg.norm(w) == {norm_w}'
                break
            
            grad_J = self.grad_J(X_transformed, y)
            
            self.J_history[iter_count] = (mse(self, X, y) 
                                          + self.lamb/y.size * norm_w**2)

            if np.linalg.norm(grad_J) <= self.tol:
                self.J_history.resize(iter_count)
                self.logger['exit'] = 'convergence'
                self.logger['exit_message'] = f'Model.__fit_gd: Convergence achieved in {iter_count} iterations.'
                break

            self.w -= self.lr * grad_J

        else:
            self.logger['exit'] = 'maxiter'
            self.logger['exit_message'] = f'Model.__fit_gd: Maximum number of {self.maxiter} iterations achieved.'

        self.logger['iter_count'] = iter_count        

    def fit(self, X, y):
        """Fit data
        """
        self.logger['last_fit_method'] = self.solver
        
        if self.solver == 'gd':
            self.__fit_gd(X, y)
        elif self.solver == 'ne':
            self.__fit_ne(X, y)
        else:
            raise RuntimeError(f'Model.fit: Unknown solver {self.solver}')
        return self

    def predict(self, X):
        """Make a prediction after fitting
        """
        X = self.__add_ones(X)
        y_pred = X @ self.w
        return y_pred

def polynomial_model(params):
    from copy import deepcopy

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    default_params = dict(
        d=2,
        maxiter=int(1e5),
        solver='gd',
        lr=1e-2,
        div_tol=1e3)
    final_params = deepcopy(default_params)
    final_params.update(params)

    d = final_params['d']
    lamb = final_params['lamb']
    maxiter = final_params['maxiter']
    solver = final_params['solver']
    lr = final_params['lr']
    div_tol = final_params['div_tol']
    fit = params['fit']

    model = make_pipeline(
        PolynomialFeatures(d, include_bias=False),
        Model(lamb=lamb, maxiter=maxiter, solver=solver, lr=lr, div_tol=div_tol))

    if fit:
        data = params['data']
        X = data['X']
        y = data['y']
        model.fit(X, y)

    return model

d = 3
lamb = 1e-3
lr = 1

pipe = manager.provide_model(
    creator_method=polynomial_model,
    creator_name='polynomial_model',
    params=dict(
        d=d,
        lamb=lamb,
        solver='gd',
        lr=lr,
        div_tol=1e3,
        fit=False
    ),
    load=False,
    save=False
)

# utils.print_header('First implementation:')
# print(pipe)

model_creator = ModelCreator(
    creator_method=polynomial_model,
    creator_name='polynomial_model',
    default_params=dict(
        d=2,
        maxiter=int(1e5),
        solver='gd',
        lr=1e-2,
        div_tol=1e3
    ),
    expand_params=False
)

pipe = manager.provide_model(
    creator_method=model_creator,
    params=dict(
        d=d,
        lamb=lamb,
        solver='gd',
        lr=lr,
        div_tol=1e3,
        fit=False
    ),
    load=False,
    save=False
)

# utils.print_header('Using ModelCreator:')
# print(pipe)

def polynomial_model2(params):
    from copy import deepcopy

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    model = make_pipeline(
        PolynomialFeatures(params['d'], include_bias=False),
        Model(**utils.extract_values(params, ('lamb', 'maxiter', 'solver', 'lr', 'div_tol')))
    )

    if params['fit']:
        data = params['data']
        X = data['X']
        y = data['y']
        model.fit(X, y)

    return model

model_creator = ModelCreator(
    creator_method=polynomial_model2,
    creator_name='polynomial_model2',
    default_params=dict(
        d=2,
        maxiter=int(1e5),
        solver='gd',
        lr=1e-2,
        div_tol=1e3
    ),
    expand_params=False
)

pipe = manager.provide_model(
    creator_method=model_creator,
    params=dict(
        d=d,
        lamb=lamb,
        solver='gd',
        lr=lr,
        div_tol=1e3,
        fit=False
    ),
    load=False,
    save=False
)

# utils.print_header('Using ModelCreator and extract_values:')
# print(pipe)

def polynomial_model3(d, lamb, maxiter, solver, lr, div_tol, fit, **kwargs):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    model = make_pipeline(
        PolynomialFeatures(d, include_bias=False),
        Model(lamb=lamb, maxiter=maxiter, solver=solver, lr=lr, div_tol=div_tol))

    if fit:
        data = params['data']
        X = data['X']
        y = data['y']
        model.fit(X, y)

    return model

model_creator = ModelCreator(
    creator_method=polynomial_model3,
    creator_name='polynomial_model3',
    default_params=dict(
        d=2,
        maxiter=int(1e5),
        solver='gd',
        lr=1e-2,
        div_tol=1e3
    ),
    expand_params=True
)

pipe = manager.provide_model(
    creator_method=model_creator,
    params=dict(
        d=d,
        lamb=lamb,
        solver='gd',
        lr=lr,
        div_tol=1e3,
        fit=False
    ),
    load=False,
    save=False
)

# utils.print_header('Using ModelCreator, extract_values and expand_params:')
# print(pipe)

pipe = manager.provide_model(
    creator_method=model_creator,
    params=dict(
        d=1,
        lamb=lamb,
        solver='gd',
        lr=lr,
        div_tol=1e3,
        fit=False
    ),
    load=True,
    save=True
)

pipe = manager.provide_model(
    creator_method=model_creator,
    params=dict(
        d=2,
        lamb=lamb,
        solver='gd',
        lr=lr,
        div_tol=1e3,
        fit=False
    ),
    load=True,
    save=True
)

pipe = manager.provide_model(
    creator_method=model_creator,
    params=dict(
        d=3,
        lamb=lamb,
        solver='gd',
        lr=lr,
        div_tol=1e3,
        fit=False
    ),
    load=True,
    save=True
)

pipe = manager.provide_model(
    creator_method=model_creator,
    params=dict(
        d=1,
        lamb=lamb,
        solver='gd',
        lr=lr,
        div_tol=1e3,
        fit=False
    ),
    load=True,
    save=True
)