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

import smpc
import sysid as sid
import helper
importlib.reload(smpc)
# %%

load_name = os.path.join('sid_results', 'building_prediction_models.pkl')
with open(load_name, "rb") as f:
    res = pickle.load(f)
    ssm = res['ssm']
    msm = res['msm']


# %%

ms_mpc_settings = smpc.base.SMPCSettings(
    prob_chance_cons=.999,
    with_cov=True,
)

ms_mpc = smpc.MultiStepSMPC(msm, ms_mpc_settings)

ms_mpc.set_objective(
    Q    = 0*np.eye(msm.n_y),
    delR = 5*np.eye(msm.n_u),
    R    = 10*np.eye(msm.n_u),
    P    = 0*np.eye(msm.n_y),
)

y = ms_mpc._y_stage
ms_mpc.set_chance_cons(expr =  y[1:], ub = 25)
ms_mpc.set_chance_cons(expr = -y[1:], ub = -18)
# y[0]-y[1] >= 1
ms_mpc.set_chance_cons(expr =  -y[0]+y[1], ub = -1)

ms_mpc.setup()

# ms_mpc.lb_opt_x['y_pred',:] = 18*np.ones(4)
ms_mpc.lb_opt_x['u_pred',:] = np.array([-6, -6, -6, -6, 10])
ms_mpc.ub_opt_x['u_pred',:] = np.array([6, 6, 6, 6, 10])

# %%

np.random.seed(99)

x0 = np.array([24, 20, 20, 20, 10]).reshape(-1,1)

sys_generator = sid.SystemGenerator(
        sys_type=sid.SystemType.BUILDING,
        sig_x=res['sigma_x'],
        sig_y=res['sigma_y'],
        case_kwargs={'state_feedback':False, 'x0': x0},
        dt=1,
    )

sys = sys_generator()
u_const = np.array([0, 0, 0, 0, 10]).reshape(-1,1)
random_input = sid.RandomInput(
    n_u = 5, 
    u_lb = u_const,
    u_ub = u_const,
    u0 = u_const,
    )

# Generate an initial sequence of measurements
sys.simulate(random_input, 3)


# %%
# Solve MPC problem once and investigate the predictions
y_list = cas.vertsplit(sys.y[-msm.data_setup.T_ini:])
u_list = cas.vertsplit(sys.u[-msm.data_setup.T_ini:])

ms_mpc.make_step(y_list, u_list)

# %% 




# %%

t_past = np.arange(-msm.data_setup.T_ini,0)
t_pred = np.arange(msm.data_setup.N)

fig, ax = plt.subplots(2,1, sharex=True)

meas_lines = ax[0].plot(t_pred, ms_mpc.res_y_pred)

ax[0].set_prop_cycle(None)
for k in range(msm.n_y):
    ax[0].fill_between(t_pred, 
                       ms_mpc.res_y_pred[:,k]+ms_mpc.cp*ms_mpc.cp*ms_mpc.res_y_std[:,k], 
                       ms_mpc.res_y_pred[:,k]-ms_mpc.cp*ms_mpc.cp*ms_mpc.res_y_std[:,k], 
                       alpha=.2)
    
ax[0].set_prop_cycle(None)
ax[0].plot(t_past, ms_mpc.res_y_past, '-x')
ax[0].axhline(18, color='k', linestyle='--')
ax[0].axvline(0, color='k', linestyle='-')
ax[0].text(-2, 18.2, '$\\leftarrow$ past')
ax[0].text(.5, 18.2, 'pred. $\\rightarrow$')

input_lines = ax[1].step(t_pred[:-1],ms_mpc.res_u_pred[:-1,:4], where='post')
ax[1].set_prop_cycle(None)
ax[1].step(t_past,ms_mpc.res_u_past[:,], '-x', where='post')

ax[0].legend(meas_lines, ['rooms 1', 'rooms 2', 'rooms 3', 'rooms 4'], ncol=4)
ax[0].set_ylabel('temp. [°C]')
ax[1].set_ylabel('heat/cool power [kW]')
ax[-1].set_xlabel('time [h]')

fig.suptitle('Stochastic MPC open-loop prediction')

# %%

ms_mpc.read_from(sys)
sys.simulate(ms_mpc, 50)
# %%

fig, ax = plt.subplots(3,1, sharex=True)
meas_lines = ax[0].plot(sys.time, sys.y)
input_lines = ax[1].step(sys.time, sys.u[:,:4])
amb_temp = ax[2].step(sys.time, sys.x[:,4])
ax[0].axhline(18, color='k', linestyle='--')
ax[0].legend(meas_lines, ['rooms 1', 'rooms 2', 'rooms 3', 'rooms 4'], ncol=4)

ax[0].set_ylabel('temp. [°C]')
ax[1].set_ylabel('heat/cool\n power [kW]')
ax[2].set_ylabel('amb. temp. [°C]')
ax[2].set_xlabel('time [h]')
fig.align_ylabels()
plt.show(block=True)

# %%
fig, ax = plt.subplots(1,1, figsize=(4,4))

ax.plot(sys.y[:,1], sys.y[:,0], label='y(t)')
ax.plot(ms_mpc.res_y_pred[:,1], ms_mpc.res_y_pred[:,0], label='y_pred(t)')

cons_1 = np.linspace(17, 20, 5)
ax.plot(cons_1, cons_1+1, color='k', linestyle='--', label='constraint')
ax.axvline(18, color='k', linestyle='--')
ax.set_xlabel('T2 [°C]')
ax.set_ylabel('T1 [°C]')
ax.set_ylim(19, 21)
ax.set_xlim(17.5, 19.5)

cov0 = ms_mpc.opt_aux_num['Sigma_y_pred'][:2,:2].full()
covN = ms_mpc.opt_aux_num['Sigma_y_pred'][-4:-2,-4:-2].full()
e1=helper.plot_cov_as_ellipse(ms_mpc.res_y_pred[0,1], ms_mpc.res_y_pred[0,0], cov0, ax=ax, n_std=ms_mpc.cp, edgecolor=colors[0], facecolor=colors[0], alpha=0.2)
e2=helper.plot_cov_as_ellipse(ms_mpc.res_y_pred[-1,1], ms_mpc.res_y_pred[-1,0], covN, ax=ax, n_std=ms_mpc.cp, edgecolor=colors[1], facecolor=colors[1], alpha=0.2)
e1.set_label('covariance at t=0')
e2.set_label('covariance at t=N')
ax.legend()


# %%
