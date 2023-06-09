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
from matplotlib.animation import FuncAnimation, ImageMagickFileWriter
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

state_feedback = True

if state_feedback:
    load_name = os.path.join('sid_results', 'cstr_prediction_model_state_feedback.pkl')
else:
    load_name = os.path.join('sid_results', 'cstr_prediction_model_output_feedback.pkl')

with open(load_name, "rb") as f:
    res = pickle.load(f)
    ssm = res['ssm']
    msm = res['msm']


# %%


T_R_ub = 135
T_R_lb = 120


def get_controller( model: Union[sid.MultistepModel, sid.StateSpaceModel]) -> Union[smpc.MultiStepSMPC, smpc.StateSpaceSMPC]:

    smpc_settings = smpc.base.SMPCSettings(
        prob_chance_cons=.999,
        with_cov=True,
    )

    if isinstance(model, sid.StateSpaceModel):
        controller = smpc.StateSpaceSMPC(model, smpc_settings)
    elif isinstance(model, sid.MultistepModel):
        controller = smpc.MultiStepSMPC(model, smpc_settings)
    else:
        raise ValueError('Unknown model type')
    

    y = controller._y_stage

    if state_feedback:
        controller.set_objective(
            Q    = 0*np.diag([0, 1, 0, 0]),
            c    = np.array([0, -1, 0, 0]).reshape(-1,1),
            delR =1*np.diag([1e-4, 1e-4]),
        )
        controller.set_chance_cons(expr =  y[2], ub = T_R_ub)
    else:
        controller.set_objective(
            Q    = 0*np.diag([1, 0]),
            c    = np.array([-1, 0]).reshape(-1,1),
            delR =1*np.diag([1e-4, 1e-4]),
        )
        controller.set_chance_cons(expr =  y[1], ub = T_R_ub)


    controller.setup()

    # controller.opt_p_num['y_set',:] = np.array([1.0, 1.0, 120, 120]).reshape(-1,1)

    controller.lb_opt_x['u_pred',:] = cstr.CSTR_BOUNDS['u_lb']
    controller.ub_opt_x['u_pred',:] = cstr.CSTR_BOUNDS['u_ub']

    return controller


ss_mpc = get_controller(ssm)
ms_mpc = get_controller(msm)

# %%

def get_prepared_sys() -> sid.system:
    """
    Initialize the system and generate an initial sequence of measurements
    """
    np.random.seed(99)

    C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
    C_b_0 = 0.8 # This is the controlled variable [mol/l]
    T_R_0 = 125 #[C]
    T_K_0 = 125.0 #[C]
    x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

    sys_generator = sid.SystemGenerator(
        sys_type=sid.SystemType.CSTR,
        dt= cstr.T_STEP_CSTR,
        case_kwargs = {'x0': x0, 'state_feedback': state_feedback}
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

    return sys

ss_sys = get_prepared_sys()
ms_sys = get_prepared_sys()


# %%
# Solve MPC problem once and investigate the predictions
y_list = cas.vertsplit(ss_sys.y[-msm.data_setup.T_ini:])
u_list = cas.vertsplit(ss_sys.u[-msm.data_setup.T_ini:])

ms_mpc.make_step(y_list, u_list)
ss_mpc.make_step(y_list, u_list)

def open_loop_pred(sys, controller):
    seq_input = sid.InputFromSequence(
        controller.res_u_pred
    )
    sys.simulate(seq_input, msm.data_setup.N)

open_loop_pred(ss_sys, ss_mpc)
open_loop_pred(ms_sys, ms_mpc)

# %%


if state_feedback:
    figsize = (8, 8)
else:
    figsize = (8, 5)

fig, ax = plt.subplots(msm.n_y + msm.n_u, 1, sharex=True, figsize=figsize)

def plot_open_loop(ax, controller, model, sys, color):
    t_past = np.arange(-model.data_setup.T_ini,0)*cstr.T_STEP_CSTR
    t_pred = np.arange(model.data_setup.N)*cstr.T_STEP_CSTR

    t_true = sys.time + t_past[0]
    
    ax[0].set_prop_cycle(None)
    for k in range(model.n_y):
        meas_lines = ax[k].plot(t_pred, controller.res_y_pred[:,k], '--', label='pred.', color=color)
        ax[k].fill_between(t_pred, 
                        controller.res_y_pred[:,k]+controller.cp*controller.res_y_std[:,k], 
                        controller.res_y_pred[:,k]-controller.cp*controller.res_y_std[:,k], 
                        alpha=.4, label='pred. std.', color=color)
        
        ax[k].plot(t_true, sys.y[:,k], '-', label='true (a posteriori eval.)', color=color)
        
        ax[k].set_prop_cycle(None)
        ax[k].plot(t_past, controller.res_y_past[:,k], '-x', color=color)

    for i in range(msm.n_u):
        ax[k+i+1].step(t_pred[:-1],controller.res_u_pred[:-1,i], where='post', color=color)
        ax[k+i+1].set_prop_cycle(None)
        ax[k+i+1].step(t_past,controller.res_u_past[:,i], '-x', where='post', color=color)    

    if state_feedback:
        ax[2].axhline(T_R_ub, color='r', linestyle='--')
        ax[0].set_ylabel('c_A [mol/L]')
        ax[1].set_ylabel('c_B [mol/L]')
        ax[2].set_ylabel('T_R [K]')
        ax[3].set_ylabel('T_K [K]')
    else:
        ax[1].axhline(T_R_ub, color='r', linestyle='--')
        ax[0].set_ylabel('c_B [mol/L]')
        ax[1].set_ylabel('T_R [K]')

    ax[-1].set_ylabel('Q_dot')
    ax[-2].set_ylabel('F [L/h]')
    # ax[0].legend(ncols=3, loc='upper left', bbox_to_anchor=(0,1.4))


plot_open_loop(ax, ss_mpc, ssm, ss_sys, color=colors[0])
plot_open_loop(ax, ms_mpc, msm, ms_sys, color=colors[1])

fig.align_ylabels()
fig.suptitle('Stochastic MPC open-loop prediction')


# %% [markdown]
## Closed-loop simulation
# - Reset the system and get initial sequence
# - Initialze variables to store the predictions at each time-step

# %%

N_steps_closed_loop = 50

def run_closed_loop(controller, sys, N_steps=N_steps_closed_loop):
    U_pred = []
    Y_pred = []
    Y_std_pred = []

    def save_predictions(s):
        U_pred.append(controller.res_u_pred)
        Y_pred.append(controller.res_y_pred)
        Y_std_pred.append(controller.res_y_std)

    controller.read_from(sys)
    sys.simulate(controller, N_steps, callbacks=[save_predictions])

    closed_loop_res = {
        'U_pred': U_pred,
        'Y_pred': Y_pred,
        'Y_std_pred': Y_std_pred
    }

    return closed_loop_res


ss_sys = get_prepared_sys()
ms_sys = get_prepared_sys()

closed_loop_res_ss = run_closed_loop(ss_mpc, ss_sys)
closed_loop_res_ms = run_closed_loop(ms_mpc, ms_sys)
# %%


def plot_closed_loop_frame(ax, res, model, sys, color, i = 0):
    t_pred = sys.time[i] + np.arange(model.data_setup.N+1)*cstr.T_STEP_CSTR

    Y_pred_i = np.concatenate((sys.y[i].reshape(1,-1), res['Y_pred'][i]))
    U_pred_i = np.concatenate((sys.u[i].reshape(1,-1), res['U_pred'][i]))
    Y_std_pred_i = np.concatenate((np.zeros((1,model.n_y)), res['Y_std_pred'][i]))

    for k in range(model.n_y):

        ax[k].plot(sys.time[:i], sys.y[:i,k], '-', label='measured', color=color)
        ax[k].set_prop_cycle(None)
        ax[k].plot(t_pred, Y_pred_i[:,k], '--', label='predicted', color=color)
        ax[k].fill_between(t_pred, 
                        Y_pred_i[:,k]+ms_mpc.cp*Y_std_pred_i[:,k], 
                        Y_pred_i[:,k]-ms_mpc.cp*Y_std_pred_i[:,k], 
                        alpha=.3, label=r'pred. $\pm c_p\sigma$', color=color)

    for j in range(model.n_u):
        ax[k+j+1].step(sys.time[:i],sys.u[:i,j], where='post', color=color)
        ax[k+j+1].set_prop_cycle(None)
        ax[k+j+1].step(t_pred,U_pred_i[:,j], where='post' , linestyle='--', color=color)

def update(i):
    for ax_i in ax:
        ax_i.clear()

    ax[0].plot([], [], '-', label='SS-SMPC', color='w')
    plot_closed_loop_frame(ax, closed_loop_res_ss, ssm, ss_sys, colors[0], i)
    ax[0].plot([], [], '-', label='MS-SMPC', color='w')
    plot_closed_loop_frame(ax, closed_loop_res_ms, msm, ms_sys, colors[1], i)

    if state_feedback:
        ax[2].axhline(T_R_ub, color='r', linestyle='--')
        ax[0].set_ylabel('c_A [mol/L]')
        ax[1].set_ylabel('c_B [mol/L]')
        ax[2].set_ylabel('T_R [K]')
        ax[3].set_ylabel('T_K [K]')
    else:
        ax[1].axhline(T_R_ub, color='r', linestyle='--')
        ax[0].set_ylabel('c_B [mol/L]')
        ax[1].set_ylabel('T_R [K]')

    ax[-1].set_ylabel('Q_dot')
    ax[-2].set_ylabel('F [L/h]')
    ax[-1].set_xlabel('time [h]')
    ax[0].legend(ncols=2, loc='upper left', bbox_to_anchor=(0,2.2), framealpha=1)

    fig.align_ylabels()
    fig.tight_layout()

fig, ax = plt.subplots(msm.n_y + msm.n_u, 1, sharex=True, figsize=figsize)

anim = FuncAnimation(fig, update, frames=N_steps_closed_loop, interval=500, repeat=True)
writer = ImageMagickFileWriter(fps=2)
anim.save('closed_loop_simulation_state_feedback.gif', writer=writer)
# update(49)

plt.show(block=True)
# %%
