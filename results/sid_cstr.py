
# %%
import numpy as np
import scipy
import sys
import os
import pickle
from typing import Union, List, Dict, Tuple, Optional, Callable
import casadi as cas
import pdb
import matplotlib.pyplot as plt
import importlib
import pathlib

# Get colors
import matplotlib as mpl
colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

sys.path.append(os.path.join('..'))

import blrsmpc.sysid.sysid as sid
from blrsmpc.system import cstr

importlib.reload(sid)

# %%

cstr_model = cstr.get_CSTR_model()
cstr_sim = cstr.get_CSTR_simulator(cstr_model, cstr.T_STEP_CSTR)


# %%

settings = {
    'N': 20,
    'T_ini': 1,
    'train_samples': 800,
    'test_samples': 50, 
    'state_feedback': True,
}

sys_generator = sid.SystemGenerator(
    sys_type=sid.SystemType.CSTR,
    dt= cstr.T_STEP_CSTR,
    case_kwargs={'state_feedback': settings['state_feedback']},
)

data_train_setup = sid.DataGeneratorSetup(
    T_ini=settings['T_ini'],
    N=settings['N'],
    n_samples=settings['train_samples'],
)
data_test_setup = sid.DataGeneratorSetup(
    T_ini=settings['T_ini'],
    N=settings['N'],
    n_samples=settings['test_samples'],
)

# Class that generates a pseudo-random input signal
random_input = sid.RandomInput(
    n_u=2, 
    u_lb = cstr.CSTR_BOUNDS['u_lb'],
    u_ub = cstr.CSTR_BOUNDS['u_ub'],
    switch_prob=.6
)

data_train = sid.DataGenerator(sys_generator, data_train_setup, random_input)
data_test  = sid.DataGenerator(sys_generator, data_test_setup, random_input)
print(f'Number of inputs: {data_train.n_u}')
print(f'Number of outputs: {data_train.n_y}')


# %%


msm = sid.MultistepModel(estimate_covariance=True, scale_x=True, scale_y=True, add_bias=False)
msm.fit(data_train)
# %%
importlib.reload(sid)
ssm = sid.StateSpaceModel(estimate_covariance=True, scale_x=True, scale_y=True, add_bias=False)
ssm.fit(data_train)

# %%
def lpd(y_true, y_pred, y_pred_std):
    return -0.5 * np.log(2*np.pi*y_pred_std**2) - 0.5 * (y_true - y_pred)**2 / y_pred_std**2


# %%

n_traj = 5
n_sig = 3

fig, ax = plt.subplots(data_train.n_y, n_traj, sharex=True, sharey='row', figsize=(10, 10))

for k in range(n_traj):

    test_case = k

    y_msm_pred, y_msm_pred_std = msm.predict(data_test.M[:,[test_case]].T, uncert_type="std", with_noise_variance=True)
    y_msm_pred = y_msm_pred.reshape(-1, data_test.n_y)
    y_msm_pred_std = y_msm_pred_std.reshape(-1, data_test.n_y)
    
    y_ssm_pred, y_ssm_pred_std = ssm.predict_sequence(data_test.M[:,[test_case]], with_noise_variance=True)
    y_ssm_pred = y_ssm_pred.reshape(-1, data_test.n_y)
    y_ssm_pred_std = y_ssm_pred_std.reshape(-1, data_test.n_y)

    # y_ssm_pred_var = ssm.predict_sequence_arx_02(data_test.M[:,[test_case]])[1:, :]

    y_true  = data_test.Y_N[:,test_case].reshape(-1, data_test.n_y)

    t = np.arange(y_true.shape[0]) * cstr.T_STEP_CSTR


    for i in range(data_train.n_y):
        ax[i,k].plot(t, y_msm_pred[:,i], label='MSM')
        ax[i,k].plot(t, y_ssm_pred[:,i], label='SSM')
        # ax[i,k].plot(t, y_ssm_pred_var[:,i], label='SSM')
        ax[i,k].plot(t, y_true[:,i], label='True', color='k')
        ax[i,k].fill_between(t, y_msm_pred[:,i] - n_sig*y_msm_pred_std[:,i], y_msm_pred[:,i] + n_sig*y_msm_pred_std[:,i], alpha=.5)
        ax[i,k].fill_between(t, y_ssm_pred[:,i] - n_sig*y_ssm_pred_std[:,i], y_ssm_pred[:,i] + n_sig*y_ssm_pred_std[:,i], alpha=.5)

ax[0,0].legend()

fig.tight_layout(pad=0.5)

# ax[0,0].set_title('c_B')
# ax[0,1].set_title('T_R')



# %%
np.diagonal(ssm.LTI.P_x, axis1=1, axis2=2)@ssm.LTI.C.T

np.diagonal(ssm.LTI.P_y, axis1=1, axis2=2)

# %%
fig  = plt.figure(figsize=(8,6))
plt.spy(msm.blr.W.T)

msm.blr.W.T[:,-1]

# %%


result_dir = os.path.join('sid_results')
save_name = "02_cstr_prediction_model.pkl"
save_name = os.path.join(result_dir, save_name)

pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

with open(save_name, "wb") as f:
    res = {'msm': msm, 'ssm': ssm}
    pickle.dump(res, f)

# %%
with open(save_name, "rb") as f:
    res = pickle.load(f)
# %%
np.linalg.eig(ssm.LTI.A)[0]
# %%
np.split(data_train.M[:,0], [data_train.setup.T_ini*data_train.n_y])
# %%
[]