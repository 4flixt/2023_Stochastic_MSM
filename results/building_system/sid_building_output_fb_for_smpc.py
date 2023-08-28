
# %% [markdown]
# # Building system identification with state-space and multi-step models for SMPC
# Data with output feedback

# ## Import packages 

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import copy
import pathlib
import pickle

sys.path.append(os.path.join('..', '..'))
import blrsmpc
from blrsmpc.sysid import sysid as sid
colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
blrsmpc.plotconfig.config_mpl(os.path.join('..', '..', 'blrsmpc', 'plotconfig', 'notation.tex'))


# %% [markdown]
# ## System generation
# - Define initial state distribution
# - Define process and measurement noise
# - Define prediction horizon
# - Define input signal
#
# Generate setup files for data generation. We compare two data-sets:
# - One with a large number of samples
# - One with a small number of samples


# %%

# Initial state mean and covariance
# Used for sampling and for the KF
sigma_x0 = np.array([3,3,3,3,5])
x0_bar = np.array([20,20,20,20,15])

def get_x0() -> np.ndarray:
    # x = [T_1 T_2 T_3 T_4 T_0]
    x0 = np.random.randn(5)*sigma_x0 + x0_bar
    x0 = x0.reshape(-1,1)
    return x0

# Process and measurement noise variances
sig_x = np.array([0.0,0.0,0.0,0.0,0.5]) 
sig_y = np.ones(4)*0.1
T_ini = 3
N = 12


sys_generator = sid.SystemGenerator(
    sys_type=sid.SystemType.BUILDING,
    sig_x=sig_x,
    sig_y=sig_y,
    dt=3600,
    case_kwargs={'state_feedback': False, 'x0': get_x0}
)

# Prepare the data set 

setup_train = sid.DataGeneratorSetup(T_ini, N, n_samples=500, sig_x=sig_x, sig_y=sig_y, dt=3600)

# Class that generates a pseudo-random input signal
random_input = sid.RandomInput(
    n_u=5, 
    u_lb = np.array([-6,-6,-6,-6, 0]).reshape(-1,1),
    u_ub = np.array([6,6,6,6,30]).reshape(-1,1),
    switch_prob=np.array([0.5, 0.5, 0.5, 0.5, 0.04]).reshape(-1,1)
    )


# %% [markdown]
# ## Generate the data with the chosen configuration
# %%
np.random.seed(99)
data_train = sid.DataGenerator(sys_generator, setup_train, random_input)

# %% [markdown]
# ### Generate the test data
# %%

# Generate the same input sequence for all test cases 
random_input_sequence = random_input.gen_sequence(N)
# Input object that returns the same input sequence
sequence_input = sid.InputFromSequence(random_input_sequence)
# Initial state (identical for all test cases) 
x0_test =np.array([20, 20.5, 19.5, 21, 10]).reshape(-1,1) 

# For all samples use a system with the same initial sequence of process and meas. noise
test_sys_ref = sid.SystemGenerator(
        sys_type=sys_generator.sys_type,
        sig_x=sig_x,
        sig_y=sig_y,
        dt=sys_generator.dt,
        P0 = np.zeros((5,5)),
        case_kwargs={
            'state_feedback': False,
            'x0': x0_test
        },
)()
test_sys_ref.simulate(sequence_input, T_ini)

def test_sys_generator():
    return copy.deepcopy(test_sys_ref)

# Test data
test_data_setup = sid.DataGeneratorSetup(
    T_ini=setup_train.T_ini,
    N=setup_train.N-setup_train.T_ini,
    n_samples=100,
)

# Generate test data
data_test = sid.DataGenerator(test_sys_generator, test_data_setup, sequence_input)

# %% [markdown]
# ## System identification with multi-step model

# %%

sid_settings = {
    'estimate_covariance': True,
    'scale_x': True,
    'scale_y': True,
    'add_bias': True,
    'type': 'mle'
}

msm = sid.MultistepModel(**sid_settings)
msm.fit(data_train)
ssm = sid.StateSpaceModel(**sid_settings)
ssm.fit(data_train)
# %% [markdown]
# ## Compare the models

# %%
y_msm_pred, y_msm_pred_std = msm.predict(data_test.M[:,[0]].T, uncert_type="std", with_noise_variance=True)
y_msm_pred = y_msm_pred.reshape(-1,data_test.n_y)
y_msm_pred_std = y_msm_pred_std.reshape(-1,data_test.n_y)
y_ssm_pred, y_ssm_pred_std = ssm.predict_sequence(data_test.M[:,[0]], with_noise_variance=True)
# %%
figsize = (blrsmpc.plotconfig.textwidth, blrsmpc.plotconfig.textwidth/2)
fig, ax = plt.subplots(data_test.n_y, 1,figsize=figsize, dpi=300, sharex=True)

t_ini = data_test.setup.T_ini

Y_pred = np.stack([sample.y[T_ini:] for sample in data_test.sim_results],axis=2)


label_list =['$T_1$ [$^\circ$C]', '$T_2$ [$^\circ$C]', '$T_3$ [$^\circ$C]', '$T_4$ [$^\circ$C]', '$T_{a}$ [$^\circ$C]'] 
t = data_test.sim_results[0].time[T_ini:]/3600
for i in range(data_test.n_y):
    ax[i].plot(t, Y_pred[:,i,:], color='k',linewidth=1, alpha=0.05)

    ax[i].plot(t, y_ssm_pred[:,i],linewidth=2, label="SSM")
    ax[i].fill_between(t, y_ssm_pred[:,i]-3*y_ssm_pred_std[:,i], y_ssm_pred[:,i]+3*y_ssm_pred_std[:,i], alpha=0.3)

    ax[i].plot(t, y_msm_pred[:,i], '--', linewidth=2, label="MSM")
    ax[i].fill_between(t, y_msm_pred[:,i]-3*y_msm_pred_std[:,i], y_msm_pred[:,i]+3*y_msm_pred_std[:,i], alpha=0.3)
    ax[i].set_ylabel(label_list[i])

ax[0].plot([],[], color='k', linewidth=1, alpha=0.2, label="Samples")
ax[-1].set_xlabel("Time [h]")

ax[0].legend()

fig.align_ylabels()
fig.tight_layout()

# %% [markdown]
# ## Export the models

# %%
result_dir = os.path.join('sid_results')
save_name = f"sid_building_tini_{t_ini}_n_{N}.pkl"
save_name = os.path.join(result_dir, save_name)

pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

with open(save_name, "wb") as f:
    res = {'msm': msm, 'ssm': ssm, 'sigma_x': sig_x, 'sigma_y': sig_y, 't_ini': T_ini, 'N': N}
    pickle.dump(res, f)
# %%
# Check if loading works
with open(save_name, "rb") as f:
    loaded = pickle.load(f)

# %%
