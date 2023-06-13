# %% [markdown]
"""
# Stochatic MPC (SMPC) for CSTR with identified multistep model and state-space model

Import the necessary packages
"""

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
import pandas as pd

# Get colors
import matplotlib as mpl
colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

sys.path.append(os.path.join('..'))

# %% [markdown]
"""
## Import the custom packages used for this project
"""

# %%

import blrsmpc
from blrsmpc import smpc
import blrsmpc.sysid.sysid as sid
from blrsmpc.system import cstr

# %% [markdown]
"""
## Load the identified system models
"""
# %%
load_name = os.path.join('sid_results', 'cstr_prediction_models.pkl')

with open(load_name, "rb") as f:
    res = pickle.load(f)
    ssm_state_fb = res['state_feedback']['ssm']
    msm_state_fb = res['state_feedback']['msm']
    ssm_output_fb = res['output_feedback']['ssm']
    msm_output_fb = res['output_feedback']['msm']


# %% [markdown]
"""
# Functions to obtain SMPC controller and evaluation tools
## SMPC controller
"""

# %%


T_R_ub = 135 #TODO: Remove and take from CSTR
T_R_lb = 120


def get_controller( model: Union[sid.MultistepModel, sid.StateSpaceModel], chance_cons: bool = True) -> blrsmpc.smpc.base.SMPCBase:

    smpc_settings = smpc.base.SMPCSettings(
        prob_chance_cons=.999,
        with_cov=True,
    )
    smpc_settings.surpress_ipopt_output()

    if isinstance(model, sid.StateSpaceModel):
        controller = smpc.StateSpaceSMPC(model, smpc_settings)
    elif isinstance(model, sid.MultistepModel):
        controller = smpc.MultiStepSMPC(model, smpc_settings)
    else:
        raise ValueError('Unknown model type')
    
    model.n_y
    

    yk = controller._y_stage
    yset = controller._y_setpoint
    uk = controller._u_stage
    up = controller._u_previous

    du = uk - up

    stage_cost = 1e-2*du[0]**2 + 1e-4*du[1]**2

    if model.n_y == 4:
        print('Model with 4 outputs (state feedback)')
        stage_cost+= -yk[1]*uk[0] # maximize c_b * F (product yield)
        T_R_ind = 2
    if model.n_y == 2:
        print('Model with 2 outputs (output feedback)')
        stage_cost += -yk[0]*uk[0] # maximize c_a * F (product yield)
        T_R_ind = 1

    stage_cost_fun = cas.Function('stage_cost', [yk, uk, up, yset], [stage_cost])
    controller.set_objective_fun(stage_cost_fun)

    if chance_cons:
        controller.set_chance_cons(expr =  yk[T_R_ind], ub = T_R_ub)

    controller.setup()

    # controller.opt_p_num['y_set',:] = np.array([1.0, 1.0, 120, 120]).reshape(-1,1)

    controller.lb_opt_x['u_pred',:] = cstr.CSTR_BOUNDS['u_lb']
    controller.ub_opt_x['u_pred',:] = cstr.CSTR_BOUNDS['u_ub']

    if not chance_cons:
        controller.ub_opt_x['y_pred', :, T_R_ind] = T_R_ub

    return controller

# %% [markdown]
"""
## Function to get a prepared system with initial sequence of measurements
"""

# %%

def get_prepared_sys(model: Union[sid.MultistepModel, sid.StateSpaceModel], x0: Optional[np.ndarray] = None) -> sid.system:
    """
    Initialize the system and generate an initial sequence of measurements
    """
    np.random.seed(99)

    if x0 is None:
        C_a_0 = 0.5 # This is the initial concentration inside the tank [mol/l]
        C_b_0 = 0.5 # This is the controlled variable [mol/l]
        T_R_0 = 120 #[C]
        T_K_0 = 120.0 #[C]
        x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

    # Check if the identified model used data with state feedback or without
    state_feedback = True if model.n_y == 4 else False

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
    sys.simulate(random_input, model.data_setup.T_ini)

    return sys

# %% [markdown]
"""
## Functions for open-loop prediction and plotting
"""

# %%

def open_loop_pred(sys: blrsmpc.system.System, controller: blrsmpc.smpc.base.SMPCBase) -> None:
    """
    Take system and controller, read the last T_ini measurements from the system and make a prediction.
    The system and controller state is changed and the function returns nothing.
    """
    y_list = cas.vertsplit(sys.y[-controller.sid_model.data_setup.T_ini:])
    u_list = cas.vertsplit(sys.u[-controller.sid_model.data_setup.T_ini:])

    controller.make_step(y_list, u_list)

    seq_input = sid.InputFromSequence(
        controller.res_u_pred
    )
    sys.simulate(seq_input, controller.sid_model.data_setup.N)

# %%
def plot_open_loop(ax, 
                   controller: blrsmpc.smpc.base.SMPCBase, 
                   sys: blrsmpc.system.System, 
                   color: str
                   ) -> None:

    n_y = controller.sid_model.n_y
    n_u = controller.sid_model.n_u

    t_past = np.arange(-controller.sid_model.data_setup.T_ini,0)*cstr.T_STEP_CSTR
    t_pred = np.arange(controller.sid_model.data_setup.N)*cstr.T_STEP_CSTR

    t_true = sys.time + t_past[0]
    
    ax[0].set_prop_cycle(None)
    for k in range(n_y):
        meas_lines = ax[k].plot(t_pred, controller.res_y_pred[:,k], '--', label='pred.', color=color)
        ax[k].fill_between(t_pred, 
                        controller.res_y_pred[:,k]+controller.cp*controller.res_y_std[:,k], 
                        controller.res_y_pred[:,k]-controller.cp*controller.res_y_std[:,k], 
                        alpha=.3, label='pred. std.', color=color)
        
        ax[k].plot(t_true, sys.y[:,k], '-', label='true', color=color)
        
        ax[k].set_prop_cycle(None)
        ax[k].plot(t_past, controller.res_y_past[:,k], '-x', label='init.', color=color)

    for i in range(n_u):
        ax[k+i+1].step(t_pred[:-1],controller.res_u_pred[:-1,i], where='post', color=color)
        ax[k+i+1].set_prop_cycle(None)
        ax[k+i+1].step(t_past,controller.res_u_past[:,i], '-x', where='post', color=color)    

    if controller.sid_model.n_y == 4:
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

def comparison_plot_open_loop(
    sys_and_controller_list: List[Tuple[blrsmpc.system.System, blrsmpc.smpc.base.SMPCBase]],
    case_names: List[str] = None,   
    ) -> Tuple[plt.Figure, plt.Axes]:

    n_y = sys_and_controller_list[0][1].sid_model.n_y
    n_u = sys_and_controller_list[0][1].sid_model.n_u

    if n_y == 4:
        figsize = (8, 8)
    else:
        figsize = (8, 5)

    fig, ax = plt.subplots(n_y + n_u, 1, sharex=True, figsize=figsize)

    for i, (sys, controller) in enumerate(sys_and_controller_list):
        if i>0:
            ax[0].plot([], [], '-', label=' ', color='w')
        ax[0].plot([], [], '-', label=case_names[i], color='w')
        plot_open_loop(ax, controller, sys, color=colors[i%len(colors)])

    ax[0].legend(loc='upper right', bbox_to_anchor=(1.25,1))

    fig.align_ylabels()
    fig.suptitle('Stochastic MPC open-loop prediction')
    # fig.tight_layout()

    return fig, ax


# %% [markdown]

"""
## Functions for closed-loop simulation
"""

# %%
def run_closed_loop(controller, sys, N_steps):
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

# %% [markdown]
"""
## Functions for plotting closed-loop simulation
"""
# %%
def plot_closed_loop(
        ax: List[plt.Axes], 
        res: Dict[str, np.ndarray], 
        controller: blrsmpc.smpc.base.SMPCBase, 
        sys: sid.system, 
        color:str, 
        i: int = 0
        ):
    
    t_pred = sys.time[i] + np.arange(controller.sid_model.data_setup.N+1)*cstr.T_STEP_CSTR

    Y_pred_i = np.concatenate((sys.y[i].reshape(1,-1), res['Y_pred'][i]))
    U_pred_i = np.concatenate((sys.u[i].reshape(1,-1), res['U_pred'][i]))
    Y_std_pred_i = np.concatenate((np.zeros((1,sys.n_y)), res['Y_std_pred'][i]))

    for k in range(sys.n_y):
        ax[k].plot(sys.time[:i+1], sys.y[:i+1,k], '-', label='measured', color=color)
        ax[k].set_prop_cycle(None)
        ax[k].plot(t_pred, Y_pred_i[:,k], '--', label='predicted', color=color)
        ax[k].fill_between(t_pred, 
                        Y_pred_i[:,k]+controller.cp*Y_std_pred_i[:,k], 
                        Y_pred_i[:,k]-controller.cp*Y_std_pred_i[:,k], 
                        alpha=.3, label=r'pred. $\pm c_p\sigma$', color=color)

    for j in range(sys.n_u):
        ax[k+j+1].step(sys.time[:i+1],sys.u[:i+1,j], where='post', color=color)
        ax[k+j+1].set_prop_cycle(None)
        ax[k+j+1].step(t_pred,U_pred_i[:,j], where='post' , linestyle='--', color=color)

# %%

class ComparisonPlotClosedLoop:
    def __init__(
            self,
            sys_list : List[blrsmpc.system.System],
            controller_list : List[blrsmpc.smpc.base.SMPCBase],
            closed_loop_res: List[Dict[str, np.ndarray]],
            case_names: List[str] = None,
            ):
        self.sys_list = sys_list
        self.n_compare = len(sys_list)

        if case_names is None:
            case_names = [None,]*len(sys_list)

        for l in [sys_list, controller_list, closed_loop_res, case_names]:
            if not isinstance(l, list):
                raise TypeError('All arguments must be lists')
            if len(l) != self.n_compare:
                raise ValueError('All lists must have the same length')

        self.controller_list = controller_list
        self.closed_loop_res = closed_loop_res
        self.case_names = case_names        

        for i, sys_i in enumerate(sys_list):
            if i==0:
                self.n_u = sys_i.n_u
                self.n_y = sys_i.n_y
            
            assert self.n_u == sys_i.n_u, 'All systems must have the same number of inputs'
            assert self.n_y == sys_i.n_y, 'All systems must have the same number of outputs'

        if self.n_y == 4:
            figsize = (8, 8)
        else:
            figsize = (8, 5)

        self.fig, self.ax = plt.subplots(self.n_u+self.n_y, 1, figsize=figsize)

    def draw_frame(self, i):
    
        ax = self.ax
        
        for ax_i in ax:
            ax_i.clear()

        for k in range(self.n_compare):
            ax[0].plot([], [], '-', label=self.case_names[k], color='w')
            plot_closed_loop(ax, self.closed_loop_res[k], self.controller_list[k], self.sys_list[k], colors[k], i)

        if self.n_y == 4:
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

        self.fig.align_ylabels()
        self.fig.tight_layout()

# %% [markdown]
"""
## Functions for key performance indicators (KPIs)
"""
# %%
def get_KPI(sys, prefix:str = 'Result'):
    if sys.n_y == 4:
        cb = sys.y[:,1]
        TR = sys.y[:,2]
        F  = sys.u[:,0]
    else:
        cb = sys.y[:,0]
        TR = sys.y[:,1]
        F  = sys.u[:,0]

    prod = np.sum(cb*F*cstr.T_STEP_CSTR)
    cons_viol = np.max(np.maximum(0, TR-T_R_ub))

    # print(f'{prefix}: {prod:.2f} mol with max {cons_viol:.2f} K constraint violation')

    return prod, cons_viol

# %% [markdown]
"""
# Evaluation
## Open-loop prediction
### State feedback
"""

# %%

ss_mpc_state_fb = get_controller(ssm_state_fb, chance_cons=True)
ms_mpc_state_fb = get_controller(msm_state_fb, chance_cons=True)

ss_sys = get_prepared_sys(ssm_state_fb)
ms_sys = get_prepared_sys(msm_state_fb)

open_loop_pred(ss_sys, ss_mpc_state_fb)
open_loop_pred(ms_sys, ms_mpc_state_fb)

comparison_plot_open_loop(
    [(ms_sys, ms_mpc_state_fb), (ss_sys, ss_mpc_state_fb)], case_names=['MSM', 'SSM']
)

# %% [markdown]
"""
### Output feedback
"""
# %%

ss_mpc_output_fb = get_controller(ssm_output_fb, chance_cons=True)
ms_mpc_output_fb = get_controller(msm_output_fb, chance_cons=True)

ss_sys = get_prepared_sys(ssm_output_fb)
ms_sys = get_prepared_sys(msm_output_fb)

open_loop_pred(ss_sys, ss_mpc_output_fb)
open_loop_pred(ms_sys, ms_mpc_output_fb)

comparison_plot_open_loop(
    [(ms_sys, ms_mpc_output_fb), (ss_sys, ss_mpc_output_fb)], case_names=['MSM', 'SSM']
)
# %% [markdown]
"""
## Closed-loop simulation
### State feedback
"""

# %%

N_steps_closed_loop = 50
ss_sys_state_fb = get_prepared_sys(ssm_state_fb)
ms_sys_state_fb = get_prepared_sys(msm_state_fb)

cl_ss_state_fb = run_closed_loop(ss_mpc_state_fb, ss_sys_state_fb, N_steps_closed_loop)
cl_ms_state_fb = run_closed_loop(ms_mpc_state_fb, ms_sys_state_fb, N_steps_closed_loop)


# %%

comp_cl_plot_state_fb = ComparisonPlotClosedLoop(
    sys_list = [ms_sys_state_fb, ss_sys_state_fb],
    controller_list = [ms_mpc_state_fb, ss_mpc_state_fb],
    closed_loop_res = [cl_ms_state_fb, cl_ss_state_fb],
    case_names=['MSM', 'SSM']
)

comp_cl_plot_state_fb.draw_frame(49)

# anim = FuncAnimation(fig, update_closed_loop_frame, frames=N_steps_closed_loop, interval=500, repeat=True)
# writer = ImageMagickFileWriter(fps=2)
# # anim.save('02_closed_loop_simulation_state_feedback.gif', writer=writer)
# update_closed_loop_frame(49)

# plt.show(block=True)

# %%
get_KPI(ms_sys_state_fb, 'MSM (state feedback)')
get_KPI(ss_sys_state_fb, 'SSM (state feedback)')

# %% [markdown]
"""
## Closed-loop simulation
### Output feedback
"""

# %%

N_steps_closed_loop = 50
ss_sys_output_fb = get_prepared_sys(ssm_output_fb)
ms_sys_output_fb = get_prepared_sys(msm_output_fb)

cl_res_ss_output_fb = run_closed_loop(ss_mpc_output_fb, ss_sys_output_fb, N_steps_closed_loop)
cl_res_ms_output_fb = run_closed_loop(ms_mpc_output_fb, ms_sys_output_fb, N_steps_closed_loop)


# %%

comp_cl_plot_output_fb = ComparisonPlotClosedLoop(
    sys_list = [ms_sys_output_fb, ss_sys_output_fb],
    controller_list = [ms_mpc_output_fb, ss_mpc_output_fb],
    closed_loop_res = [cl_res_ms_output_fb, cl_res_ss_output_fb],
)

comp_cl_plot_output_fb.draw_frame(49)

# %%
# %%
get_KPI(ms_sys_output_fb, 'MSM (state feedback)')
get_KPI(ss_sys_output_fb, 'SSM (state feedback)')

# %% [markdown]
"""
## Meta analysis closed-loop
"""

def run_meta_analysis(x0_list: List[np.ndarray], controller: Union[smpc.MultiStepSMPC, smpc.StateSpaceSMPC], N_steps: int):
    res = []

    for x0_k in x0_list:
        sys_k = get_prepared_sys(controller.sid_model, x0=x0_k)
        _ = run_closed_loop(controller, sys_k, N_steps)

        res.append(sys_k)

    return res

# %%
n_cases = 5
x0_test = np.random.uniform(cstr.CSTR_SAMPLE_BOUNDS['x_lb'], cstr.CSTR_SAMPLE_BOUNDS['x_ub'], size=(4, n_cases))
x0_test = np.split(x0_test, n_cases, axis=1)

meta_msm_output_fb = run_meta_analysis(x0_test, ms_mpc_output_fb, N_steps_closed_loop)
meta_ssm_output_fb = run_meta_analysis(x0_test, ss_mpc_output_fb, N_steps_closed_loop)
meta_msm_state_fb = run_meta_analysis(x0_test, ms_mpc_state_fb, N_steps_closed_loop)
meta_ssm_state_fb = run_meta_analysis(x0_test, ss_mpc_state_fb, N_steps_closed_loop)


# %%
df_output_fb_ssm = pd.DataFrame(map(get_KPI, meta_ssm_output_fb), columns=['product [mol]','max. cons. viol [K]'])
df_output_fb_msm = pd.DataFrame(map(get_KPI, meta_msm_output_fb), columns=['product [mol]','max. cons. viol [K]'])
df_output_fb = pd.concat([df_output_fb_ssm, df_output_fb_msm], keys = ['SSM', 'MSM'], axis=1)


df_state_fb_msm = pd.DataFrame(map(get_KPI, meta_msm_state_fb), columns=['product [mol]','max. cons. viol [K]'])
df_state_fb_ssm = pd.DataFrame(map(get_KPI, meta_ssm_state_fb), columns=['product [mol]','max. cons. viol [K]'])
df_state_fb = pd.concat([df_state_fb_ssm, df_state_fb_msm], keys = ['SSM', 'MSM'], axis=1)

df = pd.concat([df_output_fb, df_state_fb], keys = ['Output feedback', 'State feedback'], axis=1)
df
