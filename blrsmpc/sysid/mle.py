import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Union
from sklearn.preprocessing import StandardScaler
import casadi as cas


@dataclass
class MLESettings:
    """
    Settings for the MLE algorithm.
    """
    
    scale_x: bool = False
    """
    If True, the input data is scaled to have zero mean and unit variance.
    """
    scale_y: bool = False
    """
    If True, the output data is scaled to have zero mean and unit variance.
    """

    add_bias: bool = True
    """
    If True, a bias term is added to the input data.
    """

    estimate_covariance: bool = True
    """
    If True, the covariance matrix of the noise is estimated. Otherwise only the variance is estimated.
    """

class MLE:
    def __init__(self,
                scale_x: bool = False,
                scale_y: bool = False,
                add_bias: bool = True,
                estimate_covariance: bool = True,
                 ):
        
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.add_bias = add_bias
        self.estimate_covariance = estimate_covariance

        self.state = {
            'scale_x': scale_x,
            'scale_y': scale_y,
            'add_bias': add_bias,
            'trained': False,
            'feature_function': False,
            'estimate_covariance': estimate_covariance,
        }



    def _prepare_X(self, X: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """
        Private method to prepare input data. 
        - Scale data (optional)
        - Add bias (optional)
        - Apply feature function (optional)
        """

        n_d = X.shape[0]

        if self.scale_x:
            if fit_scaler:
                self.scaler_x = StandardScaler()
                self.scaler_x.fit(X)
            X_scaled = self.scaler_x.transform(X) # type:ignore
        else:
            X_scaled = X

        if self.add_bias:
            X_final = np.hstack((X_scaled, np.ones((n_d,1))))
        else:
            X_final = X_scaled
    
        return X_final   

    def _prepare_Y(self, Y: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        # Scale data
        if self.scale_y:
            if fit_scaler:
                self.scaler_y = StandardScaler()
                self.scaler_y.fit(Y)
            Y_scaled = self.scaler_y.transform(Y)    # type:ignore
        else:
            Y_scaled = Y

        return Y_scaled # type:ignore
    
    def _sparse_least_squares_solution(
            self, 
            X: np.ndarray, 
            Y: np.ndarray,
            w_sparsity: Optional[cas.Sparsity] = None,
            ) -> np.ndarray:

        """
        Solves the least squares problem with sparsity pattern.
        
        min_W ||Y - XW||

        If w_sparsity is None, the solution is given by

        W = inv(X.T@X)@X.T@Y

        The sparsity pattern must match the shape of W.
        """

        # Solve for W without sparsity pattern
        if w_sparsity is None:
            W = np.linalg.inv(X.T@X)@X.T@Y
            return W

        elif isinstance(w_sparsity, cas.Sparsity):
            w_sparsity = np.array(w_sparsity, dtype=bool)

        elif isinstance(w_sparsity, np.ndarray):
            w_sparsity = w_sparsity.astype(bool)

        W_shape = (X.shape[1], Y.shape[1])

        if not w_sparsity.shape == W_shape:
            raise ValueError('The shape of w_sparsity must be equal to the shape of W.')
        
        # Solve for W with known sparisty pattern
        W = np.zeros(w_sparsity.shape)

        for k,sparsity_k in enumerate(w_sparsity.T):
            Xk = X[:,sparsity_k]
            yk = Y[:,[k]]

            wk = np.linalg.inv(Xk.T@Xk)@Xk.T@yk
            W[sparsity_k,k] = wk.flatten()

        return W

    def fit(
            self, 
            X: np.ndarray, 
            Y: np.ndarray, 
            w_sparsity: Optional[cas.Sparsity] = None,
            *args, **kwargs):
        """
        Fit the model to the data.
        """
        # Prepare data
        X_final = self._prepare_X(X, fit_scaler=True)
        Y_final = self._prepare_Y(Y, fit_scaler=True)

        self.n_x = X_final.shape[1]
        self.n_y = Y_final.shape[1]

        # Number of data points and parameters
        n = X_final.shape[0]
        p = X_final.shape[1]

        self.Lambda_p_bar = X_final.T@X_final
        self.Sigma_p_bar = np.linalg.inv(self.Lambda_p_bar)

        # Estimate parameters
        self.W = self._sparse_least_squares_solution(X_final, Y_final, w_sparsity=w_sparsity)

        DY = Y_final - X_final@self.W
        # Estimate noise covariance
        if self.estimate_covariance:
            self.Sigma_e = DY.T@DY/(n-p)
        else:
            self.Sigma_e = np.diag(np.sum(DY**2/(n-p), axis=0))

        self.state['trained'] = True

    def predict(self, X: np.ndarray, uncert_type: str = 'cov', with_noise_variance: bool = False, *args, **kwargs) -> np.ndarray:
        """
        Predict the output for the given input data.
        """
        # Prepare data
        X_final = self._prepare_X(X)

        # Predict output
        Y_pred = X_final@self.W

        if self.scale_y:
            Y_pred = self.scaler_y.inverse_transform(Y_pred)

        # Number of data points
        m = X_final.shape[0]

        # Compute uncertainty
        Sigma_y_pred = X_final@self.Sigma_p_bar@X_final.T

        if with_noise_variance:
            Sigma_y_pred += np.eye(m)

        if self.scale_y:
            S_y = np.diag(self.scaler_y.scale_) 
            Sigma_e = S_y@self.Sigma_e@S_y.T
        else:
            Sigma_e = self.Sigma_e

        Sigma_y_pred = np.kron(Sigma_e, Sigma_y_pred)

        if uncert_type == 'std':
            Y_std_pred = np.sqrt(np.diag(Sigma_y_pred)).reshape(-1, self.n_y, order='F')
            return (Y_pred, Y_std_pred) 
        elif uncert_type == 'cov':
            return (Y_pred, Sigma_y_pred)


