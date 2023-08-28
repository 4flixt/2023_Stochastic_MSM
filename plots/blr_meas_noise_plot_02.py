# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
import copy
sys.path.append(os.path.join('..'))
# %%

class MLE:
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y:np.ndarray):
        self.m_train = X.shape[0]
        self.n_x = X.shape[1]

        assert self.m_train > self.n_x, 'Not enough data points to estimate the parameters.'

        self.Sigma_p = np.linalg.inv(X.T@X)
        self.W = self.Sigma_p@X.T@y
        
        dt_hat = X@self.W - y
        self.Sigma_t = (dt_hat.T@dt_hat)/(self.m_train-self.n_x)

    def predict(self, X: np.ndarray):

        m_test = X.shape[0]
        t_hat = X@self.W

        cov_t_hat = np.kron(X@self.Sigma_p@X.T+np.eye(m_test), self.Sigma_t)

        return t_hat, cov_t_hat


# %%

def test_fun(X_true, Sigma_y, Sigma_x= np.zeros((1,1)), w=3):
    # Check that Sigma_e is positive definite
    if np.linalg.eig(Sigma_y)[0].min()<0:
        raise ValueError('Sigma_e is not positive definite.')

    m = X_true.shape[0]
    v = np.random.multivariate_normal(np.zeros(1), Sigma_x, m)
    X_meas = copy.copy(X_true+v)
    y = np.array([2]).reshape(1,-1)*X_true
    e = np.random.multivariate_normal(np.zeros(1), Sigma_y,m)
    y += e


    return X_meas, X_true, y
# %%

np.random.seed(99)

m = 100
sig_x0 = 5
X = np.random.normal(0, sig_x0, m).reshape(-1, 1)

# Create covariance matrix for testing purpose
sig_x = 10
sig_y = 10
Sigma_y = np.array([[sig_y**2]])
Sigma_x = np.array([[sig_x**2]])
w = 3
# Sample the test fun with noise
X_train_meas, _, y_train = test_fun(X, Sigma_y = Sigma_y, Sigma_x=Sigma_x, w=w)
n_y = y_train.shape[1]


def get_test(sig_x0, m):
    # Create test data (with same interval to check for extrapolation)
    X = np.random.normal(0, sig_x0, m).reshape(-1, 1)
    X_test_meas, _, y_test = test_fun(X, Sigma_y=Sigma_y, Sigma_x=Sigma_x)

    X = np.linspace(-20,20, 5).reshape(-1, 1)
    X_true_meas, X_true, y_true = test_fun(X, Sigma_y = np.zeros(Sigma_y.shape))

    ind = np.argsort(X_test_meas, axis=0).flatten()
    X_test_meas = X_test_meas[ind]
    y_test = y_test[ind]

    res = {
        'X_test_meas': X_test_meas,
        'y_test': y_test,
        'X_true': X_true,
        'y_true': y_true
    }

    return res

test1 = get_test(sig_x0, m)
test2 = get_test(4*sig_x0, m)


# Fit model
mle = MLE()
mle.fit(X_train_meas, y_train)
# %%
def pred_test(res):
    y_pred, y_cov = mle.predict(res['X_test_meas'])
    y_std = np.sqrt(np.diag(y_cov)).reshape(-1,1)
    res['y_pred'] = y_pred
    res['y_std'] = y_std

pred_test(test1)
pred_test(test2)


# %%
def n_sigma(res, n):
    return (res['y_pred']+n*res['y_std']).flatten()


fig, ax = plt.subplots(1)


ax.plot(X_train_meas, y_train, '.', label='train')
ax.plot(test1['X_test_meas'], test1['y_test'], 'x', label='test (in train dist.)')
ax.plot(test2['X_test_meas'], test2['y_test'], 'x', label='test (out of train dist.)')
ax.plot(test2['X_test_meas'], test2['y_pred'], '-', label='pred.')

ax.fill_between(test2['X_test_meas'].flatten(), n_sigma(test2, -3), n_sigma(test2, 3), alpha=0.3, label='$\pm 3\sigma$ pred.') # type:ignore
ax.plot(test2['X_true'], test2['y_true'], '--', label='true', color='k')

ax.set_ylabel('$t$')

ax.legend()
ax.set_xlabel('$z$, $v$')


ax.annotate("", xy=(-sig_x0, -65), xytext=(sig_x0, -65),
            arrowprops=dict(arrowstyle="|-|"))
ax.text(0, -60, r'$\sigma_z$', ha='center', va='center')

ax.annotate("", xy=(-sig_x, -50), xytext=(sig_x, -50),
            arrowprops=dict(arrowstyle="|-|"))
ax.text(0, -45, r'$\sigma_v$', ha='center', va='center')

ax.annotate("", xy=(-25, -sig_y), xytext=(-25, sig_y),
            arrowprops=dict(arrowstyle="|-|"))
ax.text(-23, 0, r'$\sigma_t$', ha='center', va='center')

# ax[0].annotate('', xy=(X_test_meas[-1], -3*sig_e_theo+ y_pred[-1]), xytext=(X_test_meas[-1], 3*sig_e_theo+ y_pred[-1]),
#             arrowprops=dict(arrowstyle="|-|"))

# ax[0].text(X_test_meas[-1], -3*sig_e_theo+ y_pred[-1]-5, r'$3\hat\sigma_t$', ha='center', va='center')
# %%
