# %% [markdown]
# # System identifcation of the non-linear CSTR model 
# Identification with multistep model (msm) and state-space model (ssm). 
# %%
import numpy as np
import scipy
import pandas as pd
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

# %% [markdown]
# ## Import the custom packages used for this project

# %%
import blrsmpc.sysid.sysid as sid
from blrsmpc.system import cstr

importlib.reload(sid)

# %% [markdown]
# # Create functions for the system identification and evaluation pipeline
# ## Function to generate data for system identification
# %%

def get_train_test_data(state_feedback: bool = False):
    np.random.seed(99)

    if state_feedback:
        T_ini = 1
    else:
        T_ini = 3

    settings = {
        'N': 20,
        'T_ini': T_ini,
        'train_samples': 800,
        'test_samples': 50, 
        'state_feedback': state_feedback,
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

    return data_train, data_test



# %%
def lpd(y_true, y_pred, Sigma_y):
    dy = y_true - y_pred
    out = -.5*np.prod(np.linalg.slogdet(2*np.pi*Sigma_y))-.5 * dy.T @ np.linalg.inv(Sigma_y) @ dy

    return out

# %%
def plot_results(msm: sid.MultistepModel, ssm: sid.StateSpaceModel, data_test: sid.DataGenerator, state_feedback: bool):
    n_traj = 4
    n_sig = 3

    if state_feedback:
        figsize = (8, 8)
    else:
        figsize = (8, 4)

    fig, ax = plt.subplots(data_test.n_y, n_traj, sharex=True, sharey='row', figsize=figsize)

    for k in range(n_traj):

        test_case = k

        y_msm_pred_vec, y_msm_pred_cov = msm.predict(data_test.M[:,[test_case]].T, uncert_type="cov", with_noise_variance=True)
        y_msm_pred = y_msm_pred_vec.reshape(-1, data_test.n_y)
        y_msm_pred_std = np.sqrt(np.diag(y_msm_pred_cov)).reshape(-1, data_test.n_y)

        y_ssm_pred_vec, y_ssm_pred_cov = ssm.predict_sequence(data_test.M[:,[test_case]], with_noise_variance=True, uncert_type="cov")
        y_ssm_pred = y_ssm_pred_vec.reshape(-1, data_test.n_y)
        y_ssm_pred_std = np.sqrt(np.diag(y_ssm_pred_cov)).reshape(-1, data_test.n_y)
    

        y_true_vec  = data_test.Y_N[:,test_case]
        y_true = y_true_vec.reshape(-1, data_test.n_y)

        lpd_msm = lpd(y_true.reshape(-1,1), y_msm_pred.reshape(-1,1), y_msm_pred_cov)
        lpd_ssm = lpd(y_true.reshape(-1,1), y_ssm_pred.reshape(-1,1), y_ssm_pred_cov)

        t = np.arange(y_true.shape[0]) * cstr.T_STEP_CSTR

        ax[-1,k].set_xlabel('Time [h]')
        ax[0,k].set_title('LPD MSM: {:.2f}\n LPD SSM: {:.2f}'.format(float(lpd_msm), float(lpd_ssm)))


        for i in range(data_test.n_y):
            ax[i,k].plot(t, y_msm_pred[:,i], label='MSM')
            ax[i,k].plot(t, y_ssm_pred[:,i], label='SSM')
            # ax[i,k].plot(t, y_ssm_pred_var[:,i], label='SSM')
            ax[i,k].plot(t, y_true[:,i], label='True', color='k')
            ax[i,k].fill_between(t, y_msm_pred[:,i] - n_sig*y_msm_pred_std[:,i], y_msm_pred[:,i] + n_sig*y_msm_pred_std[:,i], alpha=.5)
            ax[i,k].fill_between(t, y_ssm_pred[:,i] - n_sig*y_ssm_pred_std[:,i], y_ssm_pred[:,i] + n_sig*y_ssm_pred_std[:,i], alpha=.5)

    ax[0,0].legend()

    if state_feedback:
        fig.suptitle('Identification with State feedback')
        ax[0,0].set_ylabel('c_A [mol/L]')
        ax[1,0].set_ylabel('c_B [mol/L]')
        ax[2,0].set_ylabel('T_R [K]')
        ax[3,0].set_ylabel('T_K [K]')
    else:
        fig.suptitle('Identification without State feedback')
        ax[0,0].set_ylabel('c_B [mol/L]')
        ax[1,0].set_ylabel('T_R [K]')

    fig.tight_layout()

    return fig, ax

# %%
def get_lpd(data: sid.DataGenerator, model: Union[sid.MultistepModel, sid.StateSpaceModel]) -> Tuple[float, float]:
    n_samples = data.setup.n_samples

    lpd_calc = np.zeros(n_samples)
    for test_case in range(n_samples):
        if isinstance(model, sid.MultistepModel):
            y_pred_vec, y_pred_cov = model.predict(data.M[:,[test_case]].T, uncert_type="cov", with_noise_variance=True)
        elif isinstance(model, sid.StateSpaceModel):
            y_pred_vec, y_pred_cov = model.predict_sequence(data.M[:,[test_case]], uncert_type="cov", with_noise_variance=True)

        y_pred = y_pred_vec.reshape(-1, data.n_y)
        y_pred_std = np.sqrt(np.diag(y_pred_cov)).reshape(-1, data.n_y)

        y_true_vec  = data.Y_N[:,test_case]
        y_true = y_true_vec.reshape(-1, data.n_y)

        lpd_calc[test_case] = lpd(y_true.reshape(-1,1), y_pred.reshape(-1,1), y_pred_cov)

    lpd_mean = float(np.mean(lpd_calc))
    lpd_std = float(np.std(lpd_calc))

    return lpd_mean, lpd_std

# lpd_mean_ssm = get_lpd(data_test, ssm)
# lpd_mean_msm = get_lpd(data_test, msm)

# print(f'LPD MSM: {lpd_mean_msm:.2f}')
# print(f'LPD SSM: {lpd_mean_ssm:.2f}')


# %% [markdown]
# # Execute functions for system identification and evaluation
# ## State feedback
# %% 
data_train_state_fb, data_test_state_fb = get_train_test_data(state_feedback=True)

# %% [markdown]
# ### Fit the models
# %%

msm_state_fb = sid.MultistepModel(estimate_covariance=True, scale_x=True, scale_y=True, add_bias=False)
msm_state_fb.fit(data_train_state_fb)

ssm_state_fb = sid.StateSpaceModel(estimate_covariance=True, scale_x=True, scale_y=True, add_bias=False)
ssm_state_fb.fit(data_train_state_fb)

# %% [markdown]
# ### Evaluate the models on test data
fig, ax = plot_results(msm_state_fb, ssm_state_fb, data_test_state_fb, state_feedback=True)

# %% [markdown]
# ### Get LPD as Key Performance Indicator

# %%
lpd_ssm_state_fb = get_lpd(data_test_state_fb, ssm_state_fb)
lpd_msm_state_fb = get_lpd(data_test_state_fb, msm_state_fb)
print(f'LPD MSM: {lpd_msm_state_fb[0]:.2f}+-{lpd_msm_state_fb[1]:.2f}')
print(f'LPD SSM: {lpd_ssm_state_fb[0]:.2f}+-{lpd_ssm_state_fb[1]:.2f}')

# %% [markdown]
# ## Output feedback
# %% 
data_train_output_fb, data_test_output_fb = get_train_test_data(state_feedback=False)

# %% [markdown]
# ### Fit the models
# %%

msm_output_fb = sid.MultistepModel(estimate_covariance=True, scale_x=True, scale_y=True, add_bias=False)
msm_output_fb.fit(data_train_output_fb)

ssm_output_fb = sid.StateSpaceModel(estimate_covariance=True, scale_x=True, scale_y=True, add_bias=False)
ssm_output_fb.fit(data_train_output_fb)

# %% [markdown]
# ### Evaluate the models on test data
fig, ax = plot_results(msm_output_fb, ssm_output_fb, data_test_output_fb, state_feedback=False)

# %% [markdown]
# ### Get LPD as Key Performance Indicator

# %%
lpd_ssm_output_fb = get_lpd(data_test_output_fb, ssm_output_fb)
lpd_msm_output_fb = get_lpd(data_test_output_fb, msm_output_fb)
print(f'LPD MSM: {lpd_msm_output_fb[0]:.2f}+-{lpd_msm_output_fb[1]:.2f}')
print(f'LPD SSM: {lpd_ssm_output_fb[0]:.2f}+-{lpd_ssm_output_fb[1]:.2f}')

# %% [markdown]
# # Save the models and table with KPI
# %%

result_dir = os.path.join('sid_results')
save_name = 'cstr_prediction_models.pkl'
save_name = os.path.join(result_dir, save_name)

pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

with open(save_name, "wb") as f:
    res = {
        'state_feedback':{
            'msm': msm_state_fb, 
            'ssm': ssm_state_fb
            },
        'output_feedback':{
            'msm': msm_output_fb, 
            'ssm': ssm_output_fb
            }
        }
    pickle.dump(res, f)

# %%
with open(save_name, "rb") as f:
    res = pickle.load(f)
# %%
class RV:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __str__(self):
        return f'{self.mean:.2f} +- {self.std:.2f}'
    
rv = RV(1,2)

ssm_kpi = [RV(*lpd_ssm_state_fb), RV(*lpd_ssm_output_fb)]
msm_kpi = [RV(*lpd_msm_state_fb), RV(*lpd_msm_output_fb)]

df = pd.DataFrame(
    {'SSM': ssm_kpi, 'MSM': msm_kpi},
    index=['State-feedback', 'Output-feedback']
)

tex_str = df.to_latex()
tex_str = tex_str.replace('+-', r'$\pm$')

savepath = os.path.join('..', '..', '2023_CDC_L-CSS_Paper_Stochastic_MSM', 'tables')
savename = 'sid_cstr_lpd.tex'

with open(os.path.join(savepath, savename), 'w') as f:
    f.write(tex_str)
# %%
df
# %%
