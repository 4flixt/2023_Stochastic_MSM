
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
    'T_ini': 3,
    'train_samples': 500,
    'test_samples': 300, 
    'state_feedback': False,
}

sys_generator = sid.SystemGenerator(
    sys_type=sid.SystemType.CSTR,
    dt= cstr.T_STEP_CSTR,
)

data_gen_setup = sid.DataGeneratorSetup(
    T_ini=settings['T_ini'],
    N=settings['N'],
    n_samples=settings['train_samples'],
)
# Class that generates a pseudo-random input signal
random_input = sid.RandomInput(
    n_u=2, 
    u_lb = cstr.CSTR_BOUNDS['u_lb'],
    u_ub = cstr.CSTR_BOUNDS['u_ub'],
    switch_prob=.3
)

data = sid.DataGenerator(sys_generator, data_gen_setup, random_input)
# %%
msm = sid.MultistepModel(estimate_covariance=True, scale_x=True, scale_y=True, add_bias=True)
msm.fit(data)
ssm = sid.StateSpaceModel(estimate_covariance=True, add_bias=True)
ssm.fit(data)
# %%

n_traj = 10
n_sig = 3

fig, ax = plt.subplots(n_traj, data.n_y, sharex=True, figsize=(5,10))

for k in range(n_traj):

    test_case = k

    y_msm_pred, y_msm_pred_std = msm.predict(data.M[:,[test_case]].T, uncert_type="std", with_noise_variance=True)
    y_msm_pred = y_msm_pred.reshape(-1, data.n_y)
    y_msm_pred_std = y_msm_pred_std.reshape(-1, data.n_y)
    y_ssm_pred, y_ssm_pred_std = ssm.predict_sequence(data.M[:,[test_case]], with_noise_variance=True)
    y_ssm_pred = y_ssm_pred.reshape(-1, data.n_y)
    y_ssm_pred_std = y_ssm_pred_std.reshape(-1, data.n_y)
    y_true  = data.Y_N[:,test_case].reshape(-1, data.n_y)

    t = np.arange(y_true.shape[0]) * cstr.T_STEP_CSTR


    for i in range(data.n_y):
        ax[k,i].plot(t, y_msm_pred[:,i], label='MSM')
        # ax[k,i].plot(t, y_ssm_pred[:,i], label='SSM')
        ax[k,i].plot(t, y_true[:,i], label='True')
        ax[k,i].fill_between(t, y_msm_pred[:,i] - n_sig*y_msm_pred_std[:,i], y_msm_pred[:,i] + n_sig*y_msm_pred_std[:,i], alpha=.5)
        # ax[k,i].fill_between(t, y_ssm_pred[:,i] - n_sig*y_ssm_pred_std[:,i], y_ssm_pred[:,i] + n_sig*y_ssm_pred_std[:,i], alpha=.5)


ax[0,0].set_title('c_B')
ax[0,1].set_title('T_K')

# %%
np.linalg.eig(ssm.LTI.A)[0]
# %%
