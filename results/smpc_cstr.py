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
from matplotlib.animation import FuncAnimation
import importlib
import casadi as cas

# Get colors
import matplotlib as mpl
colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

sys.path.append(os.path.join('..'))

from blrsmpc import smpc
import blrsmpc.sysid.sysid as sid
from blrsmpc.system import cstr

# %%

load_name = os.path.join('sid_results', '02_cstr_prediction_model.pkl')
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
    Q    = 0*np.diag([0, 1, 0, 0]),
    c    = np.array([0, -1, 0, 0]).reshape(-1,1),
    delR =1*np.diag([1e-4, 1e-4]),
)

y = ms_mpc._y_stage

T_R_ub = 135
T_R_lb = 120
c_A_ub = 1.5

# ms_mpc.set_chance_cons(expr =  y[0], ub = c_A_ub)
ms_mpc.set_chance_cons(expr =  y[2], ub = T_R_ub)

ms_mpc.setup()

ms_mpc.opt_p_num['y_set',:] = np.array([1.0, 1.0, 120, 120]).reshape(-1,1)

ms_mpc.lb_opt_x['u_pred',:] = cstr.CSTR_BOUNDS['u_lb']
ms_mpc.ub_opt_x['u_pred',:] = cstr.CSTR_BOUNDS['u_ub']



# %%

np.random.seed(99)

# x0 = np.random.uniform(low=cstr.CSTR_BOUNDS['x_lb'], high=cstr.CSTR_BOUNDS['x_ub']).reshape(-1,1)

C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.8 # This is the controlled variable [mol/l]
T_R_0 = 125 #[C]
T_K_0 = 125.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

sys_generator = sid.SystemGenerator(
    sys_type=sid.SystemType.CSTR,
    dt= cstr.T_STEP_CSTR,
    case_kwargs = {'x0': x0}
)

sys = sys_generator()
random_input = sid.RandomInput(
    n_u=2, 
    u_lb = cstr.CSTR_BOUNDS['u_lb'],
    u_ub = cstr.CSTR_BOUNDS['u_ub'],
    switch_prob=.3
)
# Generate an initial sequence of measurements
sys.simulate(random_input, msm.data_setup.T_ini)


# %%
# Solve MPC problem once and investigate the predictions
y_list = cas.vertsplit(sys.y[-msm.data_setup.T_ini:])
u_list = cas.vertsplit(sys.u[-msm.data_setup.T_ini:])

ms_mpc.make_step(y_list, u_list)

seq_input = sid.InputFromSequence(
    ms_mpc.res_u_pred
)

sys.simulate(seq_input, msm.data_setup.N)


# %%

t_past = np.arange(-msm.data_setup.T_ini,0)*cstr.T_STEP_CSTR
t_pred = np.arange(msm.data_setup.N)*cstr.T_STEP_CSTR

t_true = sys.time + t_past[0]

fig, ax = plt.subplots(msm.n_y + msm.n_u, 1, sharex=True)

ax[2].axhline(T_R_ub, color='r', linestyle='--')

ax[0].set_prop_cycle(None)
for k in range(msm.n_y):
    meas_lines = ax[k].plot(t_pred, ms_mpc.res_y_pred[:,k])
    ax[k].fill_between(t_pred, 
                       ms_mpc.res_y_pred[:,k]+ms_mpc.cp*ms_mpc.res_y_std[:,k], 
                       ms_mpc.res_y_pred[:,k]-ms_mpc.cp*ms_mpc.res_y_std[:,k], 
                       alpha=.2)
    
    ax[k].plot(t_true, sys.y[:,k], '--')
    
    ax[k].set_prop_cycle(None)
    ax[k].plot(t_past, ms_mpc.res_y_past[:,k], '-x')

for i in range(msm.n_u):
    ax[k+i+1].step(t_pred[:-1],ms_mpc.res_u_pred[:-1,i], where='post')
    ax[k+i+1].set_prop_cycle(None)
    ax[k+i+1].step(t_past,ms_mpc.res_u_past[:,i], '-x', where='post')    


fig.suptitle('Stochastic MPC open-loop prediction')


# %% [markdown]
## Closed-loop simulation
# - Reset the system and get initial sequence
# - Initialze variables to store the predictions at each time-step

# %%

U_pred = []
Y_pred = []
Y_std_pred = []

def save_predictions(s):
    U_pred.append(ms_mpc.res_u_pred)
    Y_pred.append(ms_mpc.res_y_pred)
    Y_std_pred.append(ms_mpc.res_y_std)

sys = sys_generator()
sys.simulate(random_input, msm.data_setup.T_ini)

ms_mpc.read_from(sys)
sys.simulate(ms_mpc, 50, callbacks=[save_predictions])
# %%

fig, ax = plt.subplots(msm.n_y + msm.n_u,1, sharex=True)

def update(i):
    for ax_i in ax:
        ax_i.clear()

    t_pred = sys.time[i] + np.arange(msm.data_setup.N+1)*cstr.T_STEP_CSTR

    Y_pred_i = np.concatenate((sys.y[i].reshape(1,-1), Y_pred[i]))
    U_pred_i = np.concatenate((sys.u[i].reshape(1,-1), U_pred[i]))
    Y_std_pred_i = np.concatenate((np.zeros((1,msm.n_y)), Y_std_pred[i]))

    for k in range(msm.n_y):

        ax[k].plot(sys.time[:i], sys.y[:i,k], '-')
        ax[k].set_prop_cycle(None)
        ax[k].plot(t_pred, Y_pred_i[:,k], '--')
        ax[k].fill_between(t_pred, 
                        Y_pred_i[:,k]+ms_mpc.cp*Y_std_pred_i[:,k], 
                        Y_pred_i[:,k]-ms_mpc.cp*Y_std_pred_i[:,k], 
                        alpha=.2)

    for j in range(msm.n_u):
        ax[k+j+1].step(sys.time[:i],sys.u[:i,j], where='post')
        ax[k+j+1].set_prop_cycle(None)
        ax[k+j+1].step(t_pred,U_pred_i[:,j], where='post' , linestyle='--')

    ax[2].axhline(T_R_ub, color='r', linestyle='--')

    ax[0].set_ylabel('c_b [mol/L]')
    ax[1].set_ylabel('T_R [C]')
    ax[2].set_ylabel('F [L/min]')
    ax[3].set_ylabel('Q_dot [MW]')

    fig.align_ylabels()

frames = len(U_pred)

anim = FuncAnimation(fig, update, frames=frames, interval=500, repeat=True)


update(49)
plt.show(block=True)

# %%
msm
# %%
