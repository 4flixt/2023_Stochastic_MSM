import numpy as np
import casadi as cas
import casadi.tools as ctools
from sklearn.preprocessing import StandardScaler
import warnings
import copy
import pdb
import time
from typing import Union, List, Tuple, Dict, Optional, Callable, Any


class BayesianLinearRegression:
    """Bayesian linear regression. 
    This class provides multivariate Bayesian linear regression and mimics the interface of scikit-learn's linear regression models.
    For convenience, the :py:class:`BayesianLinearRegression` also contains (pre-) processing of the input data, 
    as these operations also need to be performed for the prediction.

    Order of optional operations on inputs:
    - scale_x
    - create features
    - add bias

    These operations are performed in:
    - :py:method:`fit`
    - :py:method:`predict`
    - :py:method:`score`

    Use ``print(BayesianLinearRegression)`` to see the current state (and results) of the model.


    """
    
    def __init__(self, 
            scale_x: bool = True, 
            scale_y: bool = True, 
            add_bias: bool = True,
            estimate_covariance: bool = True,
            *args, **kwargs
        ):
        
        # Set state from input arguments
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.add_bias = add_bias
        self.estimate_covariance = estimate_covariance

        if self.scale_x:
            self.scaler_x = StandardScaler()
        else:
            warnings.warn('Input data x is not scaled in BayesianLinearRegression. Please consider scaling the data yourself.')
        
        if self.scale_y:
            self.scaler_y = StandardScaler()
        else:
            warnings.warn('Output data y is not scaled in BayesianLinearRegression. Please consider scaling the data yourself.')
        
        self.state = {
            'scale_x': scale_x,
            'scale_y': scale_y,
            'add_bias': add_bias,
            'trained': False,
            'feature_function': False,
            'estimate_covariance': estimate_covariance,
        }

        # Introduce class attributes for type hinting.
        self.opt_x     : ctools.structure3.struct_symSX
        self.opt_p     : ctools.structure3.struct_symSX
        self.opt_x_num : ctools.structure3.DMStruct 
        self.opt_p_num : ctools.structure3.DMStruct 
        self.opt_x_lb  : ctools.structure3.DMStruct
        self.opt_x_ub  : ctools.structure3.DMStruct
        self.W : np.ndarray
        self.cas_lml_fun : Callable[[ctools.structure3.DMStruct, ctools.structure3.DMStruct], cas.DM] 


    def __repr__(self) -> str:
        return_str = 'BayesianLinearRegression \n'
        return_str += '--------------------- \n'
        return_str += 'State: \n'
        for key, value in self.state.items():
            return_str += f'- {key}: {value} \n'
        if self.state['trained']:
            return_str += f'- n_x = {self.n_x} \n'
            return_str += f'- n_phi = {self.n_phi} \n'
            return_str += f'- n_y = {self.n_y} \n'
            return_str += 'Results: \n'
            return_str += f'- train_lml: {np.round(self.train_lml, 3)} \n'
            return_str += f'- log_alpha: {np.round(self.log_alpha, 3)} \n'
            return_str += f'- Sigma_e: {np.round(self.Sigma_e, 3)} \n'
            return_str += f'- log_sigma_e: {np.round(self.log_sigma_e, 3)} \n'
        
        return return_str

    @property
    def log_sigma_e(self) -> np.ndarray:
        """ Return the log standard deviation of the noise.
        
        """
        if self.state['trained']:
            return np.log(np.sqrt(np.diag(self.Sigma_e)))
        else:
            return np.array([np.nan])

    @log_sigma_e.setter
    def log_sigma_e(self, value) -> None:
        raise AttributeError('log_sigma_e is a read-only attribute')

    @property
    def log_alpha(self) -> np.ndarray:
        """ Return the log standard deviation of the signal-to-noise ratio.
        """
        if self.state['trained']:
            return self.opt_x_num['log_alpha'].full() # type: ignore
        else:
            return np.array([np.nan])


    @log_alpha.setter
    def log_alpha(self, value) -> None:
        if self.state['trained']:
            self.opt_x_num['log_alpha'] = value # type: ignore
            self._prepare_prediction(self.opt_x_num, self.opt_p_num)
        else:
            raise RuntimeError('Model must be trained before setting log_alpha')


    def set_feature_function(self, feature_function: Callable[[np.ndarray], np.ndarray]) -> None:
        if not callable(feature_function):
            raise TypeError('feature_function must be a callable function')
        
        # Set feature function
        self.feature_function = feature_function

        # Set state
        self.state['feature_function'] = True


    def score(self, X: np.ndarray, Y: np.ndarray, scoring: str ='lml', *args, **kwargs) -> np.ndarray:
        """
        Compute the score of the model on the given data.

        Args:
            X (np.ndarray): Input data.
            Y (np.ndarray): Output data.
            scoring (str): Scoring method. Can be either ``lml``, ``mse``, ``lpd`` or ``pd``.
                - ``lml``: Log marginal likelihood.
                - ``mse``: Mean squared error.
                - ``lpd``: Log predictive density.
                - ``pd``: Predictive density.
            verbose (bool): If True, print the score.

        Returns:
            float: Score of the model.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError('X must be a numpy array')
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if not isinstance(Y, np.ndarray):
            raise TypeError('Y must be a numpy array')
        if Y.ndim != 2:
            raise ValueError('Y must be a 2D array')
    
        if scoring == 'lml':
            score = self.lml(X, Y)
        elif scoring == 'mse':
            score = self.mse(X, Y)
        elif scoring == 'lpd':
            score = self.lpd(X, Y, *args, **kwargs)
        elif scoring == 'pd':
            score = self.predictivedensity(X, Y, *args, **kwargs)
        else:
            raise ValueError(f'Scoring method {scoring} is not supported')

        return score

    def fit(self, 
            X: np.ndarray, 
            Y: np.ndarray, 
            w_sparsity: Optional[cas.Sparsity] = None,
            sigma_e_sparsity: Optional[cas.Sparsity] = None,
            nlpsol_opts: Optional[Dict[str, Any]] = {},
            **kwargs):
        """Fit model to data.
        Optional (keyword) arguments are passed to :py:meth:`init_bounds_and_guess`.

        The input is transformed depending on the configuration of the class. See :py:class:`BayesianLinearRegression` for details.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            Y (np.ndarray): Output data of shape (n_samples, n_outputs)

        """
        if not isinstance(X, np.ndarray):
            raise TypeError('X must be a numpy array')
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if not isinstance(Y, np.ndarray):
            raise TypeError('Y must be a numpy array')
        if Y.ndim != 2:
            raise ValueError('Y must be a 2D array')


        # Check and prepare data
        Phi = self._prepare_X(X, fit_scaler = True)
        Y = self._prepare_Y(Y, fit_scaler = True)

        # Store feature matrix for debugging
        self.Phi = Phi

        # Check if already initialized
        if (self.state['trained'] and 
            self.n_phi == Phi.shape[1] and 
            self.n_y == Y.shape[1]):
            pass
        else:
            # Number of features after preprocessing
            self.n_x = X.shape[1]
            self.n_y = Y.shape[1]
            self.n_phi = Phi.shape[1]
            
            self._prepare_optim(w_sparsity, sigma_e_sparsity, nlpsol_opts)
            self.init_bounds_and_guess(**kwargs)

        # Set parameters for optimization
        self.opt_p_num = self._get_opt_p(Phi, Y)

        # Solve optimization problem
        res = self.Solver(
            x0=self.opt_x_num, 
            p=self.opt_p_num, 
            lbx=self.opt_x_lb, 
            ubx=self.opt_x_ub
            )

        success = self.Solver.stats()['success']
        print('Solver success: ', success)
        self.state['trained'] = True

        self.train_lml = float(res['f'].full().flatten())
        
        # Get solution
        self.opt_x_num.master = res['x'] # type:ignore

        # Extract Sigma_e
        Sigma_e_inv_bar = self.opt_x_num['Sigma_e_inv_bar'].full()
        self.Sigma_e_inv = Sigma_e_inv_bar@Sigma_e_inv_bar.T
        self.Sigma_e = np.linalg.inv(self.Sigma_e_inv)        
        
        # Prepare prediction
        self._prepare_prediction(self.opt_x_num, self.opt_p_num)

    def _get_opt_p(self, X: np.ndarray, Y: np.ndarray) -> ctools.structure3.DMStruct:
        """
        Private method to set the parameters for the LML optimization problem. 
        """
        # Get number of data points
        n_d = X.shape[0]
        
        # Get optimization parameters
        XT_X = X.T @ X
        lam_xx, _ = np.linalg.eig(XT_X)
        XT_y = X.T @ Y
        yT_y = Y.T @ Y

        opt_p_num : ctools.structure3.DMStruct = self.opt_p(0) # type: ignore (Return of call to SXSTruct is a DMStruct)

        # Set optimization parameters to structure
        opt_p_num['n_d'] = n_d                     # number of data points
        opt_p_num['XT_X'] = XT_X                   # X^T * X 
        opt_p_num['lam_xx'] = lam_xx               # Eigenvalues of X^T * X
        opt_p_num['XT_y'] = XT_y                   # X^T * y (product of features and targets)
        opt_p_num['yT_y'] = yT_y                   # y^T * y (product of targets)

        return opt_p_num


    def predict(self, 
            X: np.ndarray, 
            uncert_type: str = 'cov',
            return_scaled: bool = False,
            with_noise_variance: bool = False,
            ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict output for given input.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            uncert_type (bool): If True, return standard deviation of prediction
            return_scaled (bool): If True, return scaled prediction (and standard deviation)
            with_noise_variance (bool): If True, return prediction with noise variance

        Returns:
            tuple[np.ndarray, np.ndarray]: Prediction and standard deviation of prediction
        """
        y_pred, Phi = self.mean(X, return_scaled = return_scaled)

        # Get number of data points
        m_t = X.shape[0]


        if uncert_type is None:
            return (y_pred, None) 

        # Uncertainty quantification
        Sigma_p = np.kron(self.Sigma_e, self.Sigma_p_bar)
        Phi_hat = np.kron(np.eye(self.n_y), Phi)
        Sigma_y = Phi_hat @ Sigma_p @ Phi_hat.T 

        if with_noise_variance:
            Sigma_y += np.kron(self.Sigma_e, np.eye(m_t))   # Add noise variance to Sigma_y

        if uncert_type == 'std':
            y_std = np.sqrt(np.diag(Sigma_y)).reshape(-1, self.n_y, order='F')
            if self.scale_y and not return_scaled:
                y_std = self.scaler_y.scale_ * y_std

            return (y_pred, y_std) 
        elif uncert_type == 'cov':
            if self.scale_y and not return_scaled:
                Sigma_y = self.scaler_y.var_ * Sigma_y

            return (y_pred, Sigma_y)


    def mean(self, X: np.ndarray, return_scaled: bool = False) -> np.ndarray:
        """
        Compute the mean of the predictive distribution.
        This method is also called from the :py:meth:`predict` method.

        Args:
            x: Input data (numpy array of shape (m, n_x))

        Returns:
            y_hat_mean: Mean of the predictive distribution (numpy array of shape (m, n_y))
            phi: Feature vector (numpy array of shape (m, n_phi))
        """
        if not isinstance(X, np.ndarray):
            raise TypeError('X must be a numpy array')
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        # Check and prepare data
        Phi = self._prepare_X(X)
        y_pred = Phi @ self.W

        if not return_scaled and self.scale_y:
            y_pred = self.scaler_y.inverse_transform(y_pred)
        
        return y_pred, Phi 
    
    def lml(self, X:np.ndarray, y: np.ndarray) -> np.ndarray:
        """Log marginal likelihood.
        
        """
        X = self._prepare_X(X)
        Y = self._prepare_Y(y)

        # Set parameters for optimization
        opt_p_test = self._get_opt_p(X, Y)

        # Evaluate lmlm function
        lml = self.cas_lml_fun(self.opt_x_num, opt_p_test).full().flatten()

        return lml

    def mse(self, X:np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mean squared error
        
        """
        y_scaled = self._prepare_Y(y)

        y_pred = self.predict(X, return_std=False)

        y_pred_scaled = self._prepare_Y(y_pred)                     # type:ignore
        return np.mean((y_pred_scaled - y_scaled)**2).flatten()     # type:ignore
    
    
    def predictivedensity(self, X: np.ndarray, y: np.ndarray, aggregate: str ='mean') -> np.ndarray:
        """Computes the predictive density instead of the log predictive density.
        See documentation of :meth:`lpd` for more information.
        """

        lpd = self.lpd(X, y, aggregate="none")
        pd = np.exp(lpd)

        if aggregate == 'none':
            return pd
        if aggregate == 'median':
            return np.median(pd).flatten()
        elif aggregate == 'mean':
            return np.mean(pd).flatten()
        else:
            raise ValueError(f'Unknown aggregation method {aggregate}') 


    def lpd(self, X: np.ndarray, y: np.ndarray, aggregate: str = 'mean') -> np.ndarray:
        """Log predictive density.
        Computes the log predictive density of the posterior distribution for the given data.
        The lpd assumes that the test data is subject noise with the same variance as the training data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            y (np.ndarray): Target data of shape (n_samples, n_targets)
            aggregate (str, optional): Aggregation method. Defaults to 'mean'. Options are
                'none': no aggregation (return lpd for each test point)
                'mean': mean over all test points
                'median': median over all test points
        
        Returns:
            np.ndarray: Log predictive density 
        """
        if not isinstance(X, np.ndarray):
            raise TypeError('X must be a numpy array')
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if not isinstance(y, np.ndarray):
            raise TypeError('y must be a numpy array')
        if y.ndim != 2:
            raise ValueError('y must be a 2D array')
        
        # Get prediction
        y_pred_scaled, Phi = self.mean(X, return_scaled=True)
        # Scale true targets 
        y_scaled = self._prepare_Y(y)
        # Difference between true and predicted targets (m, n_y)
        dY = y_scaled - y_pred_scaled # type:ignore

        Sigma_y_bar = Phi@self.Sigma_p_bar@Phi.T
        sigma_y_bar = np.diag(Sigma_y_bar)+1 # add 1 to account for noise variance 
        Sigma_y_bar = np.diag(sigma_y_bar)
        Lambda_y_bar = np.diag(1/sigma_y_bar)
        logp = -.5*self.n_y*np.log(sigma_y_bar)
        logp+= -.5*np.diag(Lambda_y_bar@dY@self.Sigma_e_inv@dY.T)
        logp+= -.5*np.linalg.slogdet(2*np.pi*self.Sigma_e)[1]

        if aggregate == 'none':
            return logp # type:ignore
        elif aggregate == 'median':
            return np.median(logp).flatten() # type:ignore
        elif aggregate == 'mean':
            return np.mean(logp).flatten()
        else:
            raise ValueError('aggregate must be either "none",  "median" or "mean"')

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
                self.scaler_x.fit(X)
            X_scaled = self.scaler_x.transform(X) # type:ignore
        else:
            X_scaled = X

        if self.state['feature_function']:
            Phi_tilde = self.feature_function(X_scaled) # type:ignore
        else:
            Phi_tilde = X_scaled

        if self.add_bias:
            Phi= np.hstack((Phi_tilde, np.ones((n_d,1))))
        else:
            Phi= Phi_tilde
    

        return Phi  # type:ignore

    def _prepare_Y(self, Y: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        # Scale data
        if self.scale_y:
            if fit_scaler:
                self.scaler_y.fit(Y)
            Y_scaled = self.scaler_y.transform(Y)    # type:ignore
        else:
            Y_scaled = Y

        return Y_scaled # type:ignore

    def _prepare_optim(self, 
            w_sparsity: Optional[cas.Sparsity] = None,
            sigma_e_sparsity: Optional[cas.Sparsity] = None,
            nlpsol_opts: Optional[Dict[str, Any]] = {},
    ) -> None:

        """Prepare optimization problem.
        
        """
        if sigma_e_sparsity:
            print("Using custom sparsity for Sigma_e")
            pass
        elif self.estimate_covariance:
            sigma_e_sparsity = cas.Sparsity.lower(self.n_y)
        else:
            sigma_e_sparsity = cas.Sparsity.diag(self.n_y)

        if w_sparsity is None:
            w_sparsity = cas.Sparsity.dense(self.n_phi, self.n_y)

        opt_x = ctools.struct_symMX([
            ctools.entry("W", shape=w_sparsity),
            ctools.entry("Sigma_e_inv_bar", shape=sigma_e_sparsity),
            ctools.entry("log_alpha")
        ])
        opt_p = ctools.struct_symMX([
            ctools.entry("n_d", shape=(1, 1)),
            ctools.entry("XT_X", shape=(self.n_phi, self.n_phi)),
            ctools.entry("lam_xx", shape=(self.n_phi, 1)),
            ctools.entry("XT_y", shape=(self.n_phi, self.n_y)),
            ctools.entry("yT_y", shape=(self.n_y, self.n_y))
        ])

        logdet_Lambda_p_bar = cas.sum1(
            cas.log(opt_p['lam_xx']+cas.exp(-opt_x['log_alpha'])))


        # J is the negative log-marginal likelihood
        J = (self.n_y/2)*self.n_phi*opt_x['log_alpha']
        J += (self.n_y*opt_p['n_d']/2)*cas.log(2*np.pi)
        J += (self.n_y/2)*logdet_Lambda_p_bar

        Sigma_e_inv_bar = opt_x['Sigma_e_inv_bar']
        Sigma_e_inv = Sigma_e_inv_bar@Sigma_e_inv_bar.T
        alpha_inv = cas.exp(-opt_x['log_alpha'])

        
        # J += .5*cas.trace(Sigma_e_inv@opt_x['W'].T@opt_p['XT_X']@opt_x['W'])
        # J += .5*cas.sum1(cas.sum2(Sigma_e_inv*(opt_x['W'].T@opt_p['XT_X']@opt_x['W'])))
        # J += -cas.trace(opt_x['W']@Sigma_e_inv@opt_p['XT_y'].T)
        # J += -cas.sum1(cas.sum2(Sigma_e_inv*(opt_p['XT_y'].T@opt_x['W'])))
        # J += .5*cas.sum1(cas.sum2(Sigma_e_inv*opt_p['yT_y']))
        sum_dy2_dw2 = (.5*(1+alpha_inv)*opt_x['W'].T@opt_p['XT_X'] - opt_p['XT_y'].T )@opt_x['W'] + .5*opt_p['yT_y']
        sum_dy2_dw2 = cas.sum1(cas.sum2(Sigma_e_inv*sum_dy2_dw2))
        J += sum_dy2_dw2

        log_sigma_e_ii = -cas.log(cas.diag(Sigma_e_inv_bar))
        J += opt_p['n_d']*cas.sum1(log_sigma_e_ii)

        # sum_dw2 = cas.sum1(cas.sum2(Sigma_w_inv*(opt_x['W'].T@opt_x['W'])))

        # J += (1/2)*sum_dw2

        # Normalize with number of data points
        J = J/opt_p['n_d']

        nlp = {'x': opt_x, 'p': opt_p, 'f': J}

        # Store optimization variables
        self.opt_x = opt_x
        self.opt_p = opt_p

        # Write solver and optimization initial guess, parameters and bounds to class.
        self.Solver = cas.nlpsol('S', 'ipopt', nlp, nlpsol_opts)    # type:ignore
        self.opt_x_num = opt_x(0)                                   # type:ignore
        self.opt_p_num = opt_p(0)                                   # type:ignore
        self.opt_x_lb  = opt_x(-np.inf)                             # type:ignore
        self.opt_x_ub  = opt_x(np.inf)                              # type:ignore

        self.cas_lml_fun = cas.Function('lml', [opt_x, opt_p], [J]) # type:ignore

    def init_bounds_and_guess(self, 
            lb_Sigma_e_inv: float = -np.inf, 
            ub_Sigma_e_inv: float = np.inf, 
            lb_log_alpha:   float =  0.0, 
            ub_log_alpha:   float =  12.0,
            Sigma_e_inv_0:  float =  0.1,
            log_alpha_0:    float =  2.0,
            **kwargs,
        ) -> None:
        """Initialize bounds and guess for optimization variables.

        Args:
            lb_log_sigma_e (float, optional): Lower bound for log(sigma_e). Defaults to -8.0.
            ub_log_sigma_e (float, optional): Upper bound for log(sigma_e). Defaults to 8.0.
            lb_log_alpha (float, optional): Lower bound for log(alpha). Defaults to 0.0.
            ub_log_alpha (float, optional): Upper bound for log(alpha). Defaults to 12.0.
        
        Returns:
            None
        """

        # Initialize bounds for optimization problem
        self.opt_x_lb['Sigma_e_inv_bar'] = lb_Sigma_e_inv
        self.opt_x_ub['Sigma_e_inv_bar'] = ub_Sigma_e_inv

        self.opt_x_lb['log_alpha'] = lb_log_alpha
        self.opt_x_ub['log_alpha'] = ub_log_alpha

        # Initialize guess for optimization problem
        self.opt_x_num['W'] = np.random.randn(self.n_phi, self.n_y)
        self.opt_x_num['Sigma_e_inv_bar'] = Sigma_e_inv_0*np.eye(self.n_y)
        self.opt_x_num['log_alpha']   = log_alpha_0

    def _prepare_prediction(self, opt_x_num, opt_p_num):
        """ Private method to prepare prediction.
        Retrieves the posterior precision matrix and the posterior weights and computes the posterior covariance matrix.
        
        """
        self.Lambda_p_bar = (opt_p_num['XT_X'] + cas.exp(-opt_x_num['log_alpha'])*np.eye(self.n_phi)).full()
        self.Sigma_p_bar = np.linalg.inv(self.Lambda_p_bar)
        self.W =opt_x_num['W'].full()



    def grid_search_alpha(self, x, y, rel_range=[0,10], scores = ['lpd'], samples=10, max_cond = 1e8, verbose=True):
        """Simple grid search to test different values of alpha.
        The search evaluates the predictive log-probability of the posterior distribution
        for a test set of data (x, y) which must not have been used for training.

        The method stops testing alpha if the condtion number of the posterior precision matrix exceeds
        ``max_cond``. 

        Args:
            x (tf.Tensor or numpy.ndarray): Input data of shape (m, n_x).
            y (tf.Tensor or numpy.ndarray): Output data of shape (m, n_y).
            rel_range (list, optional): Range of alpha values to test relative to optimal alpha*. Defaults to [0,10].
            scores (list): List of scores to evaluate. Defaults to ['lpd'].
            samples (int, optional): Number of samples to evaluate the predictive log-probability. Defaults to 10.
            max_cond (float, optional): Maximum condition number of the posterior precision matrix. Defaults to 1e8.
            verbose (bool, optional): Print training progress. Defaults to True.

        Returns:
            log_alpha_test (numpy.ndarray): Array of tested log_alpha values of shape (samples,).
            logprob_test (numpy.ndarray): Array of predictive log-probabilities for the test data of shape (samples,).
        """

        log_alpha_min = float(self.log_alpha)+rel_range[0]
        log_alpha_max = log_alpha_min +rel_range[1]
        log_alpha_opt = copy.copy(self.log_alpha)

        log_alpha_test = np.linspace(log_alpha_min, log_alpha_max, samples).astype('float32')

        results = {
            key: [] for key in scores
            }


        for k, log_alpha in enumerate(log_alpha_test):
            self.log_alpha = log_alpha

            if np.linalg.cond(self.Lambda_p_bar)>max_cond:
                log_alpha_test = log_alpha_test[:k]
                break
            else:
                for score in scores:
                    results[score].append(self.score(x, y, scoring=score))

        for score in scores:
            results[score] = np.array(results[score]).flatten() # type:ignore

        results['log_alpha'] = log_alpha_test                   # type:ignore

        # Reset to optimal alpha
        self.log_alpha = log_alpha_opt


        return results 


def test_fun(X, Sigma_e):
    # Check that Sigma_e is positive definite
    if np.linalg.eig(Sigma_e)[0].min()<0:
        raise ValueError('Sigma_e is not positive definite.')

    m = X.shape[0]
    y = -np.array([2,3]).reshape(1,-1)*X+10 
    v = np.random.multivariate_normal(np.zeros(2), Sigma_e,m)
    y += v
    return y

def test_fun_nd(W, Sigma_e, m=100):
    """
    Creates a test function with n-dimensional input and output. 
    The input and output dimension are determined by the shape of W.
    """
    n_x = W.shape[0]
    n_y = W.shape[1]

    X = np.random.randn(m, n_x)

    Y_true = X@W

    Y_meas = Y_true + np.random.multivariate_normal(np.zeros(n_y), Sigma_e, m)

    return X, Y_true, Y_meas, Sigma_e

def test_1():
    print('Running Test 1.')
    np.random.seed(99)

    blp = BayesianLinearRegression(scale_x=True, scale_y=False, add_bias=True, estimate_covariance=False)
    
    def feat_fun(x):
        return np.concatenate([x, x**2, x**3, x**4], axis=1)

    blp.set_feature_function(feat_fun)

    m = 50
    X = np.linspace(-10, 10, m).reshape(-1, 1)

    # Create covariance matrix for testing purpose
    Sigma_e = np.array([[10, 5],[5, 10]])
    # Sample the test fun with noise
    y_true = test_fun(X, Sigma_e = np.zeros((2,2)))
    y_train = test_fun(X, Sigma_e = Sigma_e)
    n_y = y_true.shape[1]
    # Create test data (with larger interval to check for extrapolation)
    X_test = np.linspace(-20, 20, 100).reshape(-1, 1)
    y_test = test_fun(X_test, Sigma_e=Sigma_e)

    # Fit model
    blp.fit(X, y_train)

    # Predict. Compute standard deviation with and without including the additive noise variance.
    # Including the additive noise variance can be useful when comparing the prediction to noisy test data.
    y_pred, y_std_with_noise = blp.predict(X_test, return_std=True, with_noise_variance=True)
    p3_sigma_with_noise = y_pred+3*y_std_with_noise
    m3_sigma_with_noise = y_pred-3*y_std_with_noise
    _, y_std= blp.predict(X_test, return_std=True)
    p3_sigma= y_pred+3*y_std
    m3_sigma= y_pred-3*y_std

    # Create figure
    fig, ax = plt.subplots(n_y,1)
    for i in range(n_y):
        ax[i].plot(X, y_true[:,i], '-', label='true')
        ax[i].plot(X, y_train[:,i], 'o', label='data')
        ax[i].plot(X_test, y_pred[:,i], '-', label='prediction')
        ax[i].fill_between(X_test.flatten(), m3_sigma_with_noise[:,i], p3_sigma_with_noise[:,i], alpha=0.5, label='std with noise') # type:ignore
        ax[i].fill_between(X_test.flatten(), m3_sigma[:,i], p3_sigma[:,i], alpha=0.5, label='std') # type:ignore

    ax[0].legend()
    ax[0].set_title('Bayesian Linear Regression with $n_y=2$ and different noise levels.')
    ax[1].set_xlabel('x')
    ax[0].set_ylabel('y_1')
    ax[1].set_ylabel('y_2')

    # Print information about the model and get some performance scores
    print(blp)

    test_lml = blp.score(X_test, y_test, scoring='lml')
    test_lpd = blp.score(X_test, y_test, scoring='lpd')
    train_lpd = blp.score(X, y_train, scoring='lpd')
    print('lpd_test = {}'.format(test_lpd))
    print('lpd_train = {}'.format(train_lpd))
    print('lml_test = {}'.format(test_lml))

    return fig


def test_2():
    """ In this test, we check that the model can handle ill-conditioned noise covariance matrices.
    We also see how well the entries of the covariance matrix are estimated depending on the name of data points.
    
    """
    print('Running Test 2.')


    blp = BayesianLinearRegression(scale_x=True, scale_y=False, add_bias=True, estimate_covariance=True)
    
    # Test for 
    n_x = 5
    n_y = 4
    W = np.random.randn(n_x, n_y)
    
    # Create noise covariance that is extremely ill-conditioned
    Sigma_e_bar = np.random.randn(n_y, n_y)
    Sigma_e_bar = np.diag(np.logspace(-3, 3, n_y))@Sigma_e_bar
    Sigma_e = Sigma_e_bar@Sigma_e_bar.T

    # Number of test points (array). We repeat each experiment 5 times.
    m_test = np.repeat(np.logspace(2, 4, 20, dtype=int), 5)
    Sigma_e_arr = np.zeros((len(m_test), n_y, n_y))

    # For all experiments:
    for k, m in enumerate(m_test):
        X, Y_true, Y_meas, Sigma_e = test_fun_nd(W, Sigma_e, m)
        blp.fit(X, Y_meas)
        Sigma_e_arr[k] = blp.Sigma_e

    fig, ax = plt.subplots(n_y, n_y, sharex=True)
    for i in range(n_y):
        for j in range(n_y):
            ax[i][j].plot(m_test, Sigma_e_arr[:,i,j], 'x', markersize=5, label='estimated')
            ax[i][j].axhline(Sigma_e[i,j], color='k', linestyle='--', label='true')
            ax[i][j].set_xlabel('m ')
            ax[i][j].set_ylabel(f'{i+1},{j+1}')
    
    fig.tight_layout()

    return fig

def test_3():
    print('Running Test 3.')
    """Test check for the log_alpha optimization."""

    blp = BayesianLinearRegression(scale_x=True, scale_y=True, add_bias=True)
    
    def feat_fun(x):
        return np.concatenate([x, x**2, x**3, np.sin(x)], axis=1)

    blp.set_feature_function(feat_fun)

    m = 50
    X = np.linspace(-10, 10, m).reshape(-1, 1)

    Sigma_e = np.array([[10, .9],[.9, .1]])
    y_true = test_fun(X, Sigma_e = np.zeros((2,2))) # (sald)
    y_train = test_fun(X, Sigma_e = Sigma_e)
    n_y = y_true.shape[1]

    X_test = np.linspace(-10, 10, 100).reshape(-1, 1)
    y_test = test_fun(X_test, Sigma_e = Sigma_e)
    blp.fit(X, y_train)
    y_pred, y_std = blp.predict(X_test, return_std=True, with_noise_variance=True)
    p3_sigma = y_pred+3*y_std
    m3_sigma = y_pred-3*y_std

    scores = ['lml', 'lpd']
    res_gs_alpha_test = blp.grid_search_alpha(X_test, y_test, rel_range=[-5, 10], samples=100, scores=scores)
    res_gs_alpha_train = blp.grid_search_alpha(X, y_train, rel_range=[-5, 10], samples=100, scores=scores)

    fig, ax = plt.subplots(4,1)
    ax[0].plot(X, y_train[:,0], 'x', label='data')
    ax[0].plot(X_test, y_test[:,0], 'o', label='test')
    ax[0].plot(X_test, y_pred[:,0], '-', label='prediction')
    ax[0].fill_between(X_test.flatten(), m3_sigma[:,0], p3_sigma[:,0], alpha=0.5, label='std') # type:ignore
    ax[1].fill_between(X_test.flatten(), m3_sigma[:,1], p3_sigma[:,1], alpha=0.5, label='std') # type:ignore
    ax[1].plot(X, y_train[:,1], 'x', label='data')
    ax[1].plot(X_test, y_test[:,1], 'o', label='test')
    ax[1].plot(X_test, y_pred[:,1], '-', label='prediction')
    ax[2].plot(res_gs_alpha_test['log_alpha'], res_gs_alpha_test['lml'], '-', label='test')
    ax[2].plot(res_gs_alpha_train['log_alpha'], res_gs_alpha_train[scores[0]], '-', label='train')
    ax[2].axvline(blp.log_alpha, color='k', linestyle='--', label='alpha_star')         # type:ignore
    ax[2].set_xlabel('log_alpha')
    ax[2].set_ylabel(scores[0])
    ax[3].plot(res_gs_alpha_test['log_alpha'], res_gs_alpha_test[scores[1]], '-', label='test')
    ax[3].plot(res_gs_alpha_train['log_alpha'], res_gs_alpha_train[scores[1]], '-', label='train')
    ax[3].axvline(blp.log_alpha, color='k', linestyle='--', label='alpha_star')        # type:ignore
    ax[3].set_xlabel('log_alpha')
    ax[3].set_ylabel(scores[1])
    ax[3].legend()

    return fig

    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    if True:
        fig1 = test_1()
    if True:
        fig2 = test_2()
    if True:
        fig3 = test_3()


    plt.show(block=True)



