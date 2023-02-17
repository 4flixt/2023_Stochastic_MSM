# %% [markdown]
"""
# Meta analyis of stochastic MPC performance for the building model

## Import packages
"""
# %%
import numpy as np
import scipy
import sys
import os
import pickle
from typing import Union, List, Dict, Tuple, Optional, Callable
from enum import Enum, auto
import casadi as cas
import pdb
import matplotlib.pyplot as plt
import importlib
import functools
import time
import random

sys.path.append(os.path.join('..'))
sys.path.append(os.path.join('..', 'plots'))

import smpc
import sysid as sid
import helper
import config_mpl
importlib.reload(config_mpl)
importlib.reload(smpc)


# %% [markdown]
"""
## Helper functions for controller setup
"""

# %%
def load_sid_results():
    load_name = os.path.join('sid_results', 'building_prediction_models.pkl')
    with open(load_name, "rb") as f:
        res = pickle.load(f)

    return res


def setup_controller(
        sid_model: Union[sid.StateSpaceModel, sid.MultistepModel], 
        smpc_settings: smpc.base.SMPCSettings
        ):

    if isinstance(sid_model, sid.StateSpaceModel):
        controller = smpc.StateSpaceSMPC(sid_model, smpc_settings)
    elif isinstance(sid_model, sid.MultistepModel):
        controller = smpc.MultiStepSMPC(sid_model, smpc_settings)

    controller.set_objective(
        Q    = 0*np.eye(sid_model.n_y),
        delR = 5*np.eye(sid_model.n_u),
        R    = 10*np.eye(sid_model.n_u),
        P    = 0*np.eye(sid_model.n_y),
    )

    y = controller._y_stage
    controller.set_chance_cons(expr =  y[1:], ub = 25)
    controller.set_chance_cons(expr = -y[1:], ub = -18)
    # y[0]-y[1] >= 1
    controller.set_chance_cons(expr =  -y[0]+y[1], ub = -1)

    controller.setup()

    # Set bounds
    controller.lb_opt_x['u_pred',:] = np.array([-6, -6, -6, -6, 10])
    controller.ub_opt_x['u_pred',:] = np.array([6, 6, 6, 6, 10])

    return controller

def get_system_for_case_study(
        sig_x, 
        sig_y, 
        t_0_rooms = np.array([23,19,19,19]), 
        t_ambient = 10, 
        link_controller: Optional[Union[smpc.StateSpaceSMPC, smpc.MultiStepSMPC]] = None,
        ):
    # Seed for deterministic initial sequence of process / measurement noise
    np.random.seed(99)
    
    x0 = np.concatenate((
        np.atleast_2d(t_0_rooms).reshape(-1,1),
        np.atleast_2d(t_ambient).reshape(-1,1),
    ), axis=0)

    sys_generator = sid.SystemGenerator(
            sys_type=sid.SystemType.BUILDING,
            sig_x=sig_x,
            sig_y=sig_y,
            case_kwargs={'state_feedback':False, 'x0': x0},
            dt=1,
        )

    sys = sys_generator()

    u_const = np.array([0,0,0,0,10]).reshape(-1,1) 
    random_input = sid.RandomInput(
        n_u=5, 
        u_lb = u_const,
        u_ub = u_const,
        u0= u_const,
        )

    # Generate an initial sequence of measurements
    sys.simulate(random_input, 3)

    np.random.seed(int(random.random()*1e6))

    if link_controller is not None:
        link_controller.read_from(sys)

    return sys

def sample_open_loop_prediction(
        controller: Union[smpc.StateSpaceSMPC, smpc.MultiStepSMPC],
        sid_res,
        n_samples: int = 50,
    ) -> sid.DataGenerator :

    open_loop_pred_input = sid.InputFromSequence(controller.res_u_pred)

    data_gen_setup = sid.DataGeneratorSetup(
        N=sid_res['msm'].data_setup.N,
        T_ini=0,
        n_samples = n_samples,
    )

    data_gen = sid.DataGenerator(
        get_sys = functools.partial(get_system_for_case_study, sid_res['sigma_x'], sid_res['sigma_y']),
        setup = data_gen_setup,
        u_fun = open_loop_pred_input,
    )

    return data_gen

def sample_closed_loop(
        controller: Union[smpc.StateSpaceSMPC, smpc.MultiStepSMPC],
        sid_res,
        n_samples: int = 10,
        N_horizon: int = 50,
    ) -> sid.DataGenerator :

    data_gen_setup = sid.DataGeneratorSetup(
        N=N_horizon,
        T_ini=0,
        n_samples = n_samples,
    )

    data_gen = sid.DataGenerator(
        get_sys = functools.partial(get_system_for_case_study, sid_res['sigma_x'], sid_res['sigma_y'], link_controller=controller),
        setup = data_gen_setup,
        u_fun = controller,
    )

    return data_gen
# %% [markdown]
"""
## Helper functions for plotting
"""
# %%

def plot_open_loop_prediction(
        controller: Union[smpc.StateSpaceSMPC, smpc.MultiStepSMPC],
        sid_model: Union[sid.StateSpaceModel, sid.MultistepModel],
        fig_ax: Optional[Tuple[plt.figure, plt.axis]] = None,
        with_annotations: bool = True,
        ) -> Tuple[plt.figure, plt.axis, Dict[str, plt.Line2D]]:
    
    t_past = np.arange(-sid_model.data_setup.T_ini,0)
    t_pred = np.arange(sid_model.data_setup.N)

    if fig_ax is None:
        fig, ax = plt.subplots(3,1, sharex=True)
    else:
        fig, ax = fig_ax

    lines = {}

    lines['y_pred'] = ax[0].plot(t_pred, controller.res_y_pred)
    ax[0].set_prop_cycle(None)
    ax[0].plot(t_pred, controller.res_y_pred+controller.cp*controller.res_y_std, '--')
    ax[0].set_prop_cycle(None)
    ax[0].plot(t_pred, controller.res_y_pred-controller.cp*controller.res_y_std, '--')
        
    ax[0].set_prop_cycle(None)
    lines['y_past'] = ax[0].plot(t_past, controller.res_y_past, '-x')
    ax[0].axhline(18, color='k', linestyle=':')
    ax[0].axvline(0, color='k', linestyle='-')

    lines['u_pred'] = ax[1].step(t_pred[:-1],controller.res_u_pred[:-1,:4], where='post')
    ax[1].set_prop_cycle(None)
    lines['u_past'] = ax[1].step(t_past,controller.res_u_past[:,:4], '-x', where='post')
    ax[1].axvline(0, color='k', linestyle='-')

    lines['t0_pred'] = ax[2].plot(t_pred[:-1],controller.res_u_pred[:-1,4])
    ax[2].set_prop_cycle(None)
    lines['t0_past'] = ax[2].plot(t_past,controller.res_u_past[:,4], '-x')
    ax[2].axvline(0, color='k', linestyle='-')

    if with_annotations:
        ax[0].legend(lines['y_pred'], ['room 1', 'room 2', 'room 3', 'room 4'], ncol=4)
        ax[0].set_ylabel('temp. [°C]')
        ax[1].set_ylabel('heat/cool\n power [kW]')
        ax[2].set_ylabel('amb. temp. [°C]')
        ax[-1].set_xlabel('time [h]')

    return fig, ax, lines

def plot_open_loop_prediction_with_samples(
    controller: Union[smpc.StateSpaceSMPC, smpc.MultiStepSMPC],
    sid_model: Union[sid.StateSpaceModel, sid.MultistepModel],
    open_loop_samples: sid.DataGenerator,
    fig_ax: Optional[Tuple[plt.figure, plt.axis]] = None,
    with_annotations: bool = True,
    ) -> Tuple[plt.figure, plt.axis, Dict[str, plt.Line2D]]:


    fig, ax, lines = plot_open_loop_prediction(controller, sid_model, fig_ax, with_annotations)

    lines['y_samples'] = []
    lines['t0_samples'] = []

    for sim_res in open_loop_samples.sim_results:
        ax[0].set_prop_cycle(None)
        lines['y_samples'].append(ax[0].plot(sim_res.time-3, sim_res.y, alpha=.1))

        ax[-1].set_prop_cycle(None)
        lines['t0_samples'].append(ax[-1].plot(sim_res.time-3, sim_res.x[:,4], alpha=.1))

    return fig, ax,lines

def plot_closed_loop_trajectory(
        closed_loop_res: sid.DataGenerator,
        fig_ax: Optional[Tuple[plt.figure, plt.axis]] = None,
        with_annotations: bool = True
        )-> Tuple[plt.figure, plt.axis, Dict[str, plt.Line2D]]:
    
    if fig_ax is None:
        fig, ax = plt.subplots(3,1, sharex=True)
    else:
        fig, ax = fig_ax

    lines = {}

    lines['y_samples'] = []
    lines['u_samples'] = []
    lines['t0_samples'] = []

    for sim_res_k in closed_loop_res.sim_results:
        lines['y_samples'].append(ax[0].plot(sim_res_k.time, sim_res_k.y, alpha=.2))
        lines['u_samples'].append(ax[1].step(sim_res_k.time, sim_res_k.u[:,:4], where='post', alpha=.2))
        lines['t0_samples'].append(ax[2].step(sim_res_k.time, sim_res_k.x[:,4], alpha=.2))

        for ax_k in ax:
            ax_k.set_prop_cycle(None)
    
    ax[0].axhline(18, color='k', linestyle='--')

    if with_annotations:
        ax[0].legend(lines['y_samples'][0], ['rooms 1', 'rooms 2', 'rooms 3', 'rooms 4'], ncol=4)

        ax[0].set_ylabel('temp. [°C]')
        ax[1].set_ylabel('heat/cool\n power [kW]')
        ax[2].set_ylabel('amb. temp. [°C]')
        ax[2].set_xlabel('time [h]')
        fig.align_ylabels()

    return fig, ax, lines

def plot_closed_loop_cons_detail(
        closed_loop_res: sid.DataGenerator,
        controller: Union[smpc.StateSpaceSMPC, smpc.MultiStepSMPC],
        fig_ax: Optional[Tuple[plt.figure, plt.axis]] = None,
        with_annotations: bool = True
        )-> Tuple[plt.figure, plt.axis]:
    
    if fig_ax is None:
        fig, ax = plt.subplots(1,1, figsize=(config_mpl.columnwidth, config_mpl.columnwidth))
    else:
        fig, ax = fig_ax


    for sim_res_k in closed_loop_ms.sim_results:
        ax.plot(sim_res_k.y[:,1], sim_res_k.y[:,0],color=config_mpl.colors[0], alpha=.1)
    ax.plot([], [], color=config_mpl.colors[0], label='y(t) (samples)', alpha=.1)
    ax.plot(controller.res_y_pred[:,1], controller.res_y_pred[:,0], color=config_mpl.colors[1], label='y_pred(t)')

    cons_1 = np.linspace(17, 20, 5)
    ax.plot(cons_1, cons_1+1, color='k', linestyle='--', label='constraint')
    ax.axvline(18, color='k', linestyle='--')

    cov0 = controller.opt_aux_num['Sigma_y_pred'][:2,:2].full()
    covN = controller.opt_aux_num['Sigma_y_pred'][-4:-2,-4:-2].full()

    e2=helper.plot_cov_as_ellipse(controller.res_y_pred[-1,1], controller.res_y_pred[-1,0], covN,
        ax=ax, n_std=controller.cp, edgecolor=config_mpl.colors[2], facecolor=config_mpl.colors[2], alpha=0.3
        )
    e1=helper.plot_cov_as_ellipse(controller.res_y_pred[0,1], controller.res_y_pred[0,0], cov0,
        ax=ax, n_std=controller.cp, edgecolor=config_mpl.colors[1], facecolor=config_mpl.colors[1], alpha=0.3
        )
    ax.set_ylim(19, 21)
    ax.set_xlim(17.5, 19.5)

    if with_annotations:
        e1.set_label('covariance at t=0')
        e2.set_label('covariance at t=N')
        ax.set_xlabel('T2 [°C]')
        ax.set_ylabel('T1 [°C]')
        ax.legend()

    return fig, ax

# %% [markdown]
"""
## Evaluations

### SMPC with multi-step model
"""
# %% 
if __name__ == '__main__':

    sid_res = load_sid_results()

    smpc_settings = smpc.base.SMPCSettings(
        prob_chance_cons=.99,
        with_cov=True,
    )
    smpc_settings.surpress_ipopt_output()

    ms_smpc = setup_controller(sid_res['msm'], smpc_settings)

    test_sys = get_system_for_case_study(sid_res['sigma_x'], sid_res['sigma_y'])


    # %%

    ms_smpc.read_from(test_sys)
    # Simulate the system using the controller for n time steps
    ms_smpc(None, None)

    _ = plot_open_loop_prediction(ms_smpc, sid_res['msm'])

    # %% [markdown]
    """
    Sample the real system with this open-loop prediction
    """
    open_loop_pred_samples_ms = sample_open_loop_prediction(ms_smpc, sid_res, n_samples = 50)

    _ = plot_open_loop_prediction_with_samples(ms_smpc, sid_res['msm'], open_loop_pred_samples_ms)

    # %% [markdown]
    """
    ## Closed-loop simulation
    """
    closed_loop_ms = sample_closed_loop(ms_smpc, sid_res, n_samples = 10, N_horizon=50)


    # %%

    _ = plot_closed_loop_trajectory(closed_loop_ms)
    _ = plot_closed_loop_cons_detail(closed_loop_ms, ms_smpc)
    # %%

    # %% [markdown]
    """
    ### SMPC with state-space model
    """
    # %%
    
    sid_res = load_sid_results()

    smpc_settings = smpc.base.SMPCSettings(
        prob_chance_cons=.99,
        with_cov=True,
    )
    smpc_settings.surpress_ipopt_output()
    smpc_settings.nlp_opts['ipopt.max_iter'] = 500

    ss_smpc = setup_controller(sid_res['ssm'], smpc_settings)

    test_sys = get_system_for_case_study(sid_res['sigma_x'], sid_res['sigma_y'])

    # %%
    ss_smpc.read_from(test_sys)
    # Simulate the system using the controller for n time steps
    ss_smpc(None, None)

    _ = plot_open_loop_prediction(ss_smpc, sid_res['ssm'])


    # %% [markdown]
    """
    Sample the real system with this open-loop prediction
    """
    _ = open_loop_pred_samples_ss = sample_open_loop_prediction(ss_smpc, sid_res, n_samples = 50)

    _ = plot_open_loop_prediction_with_samples(ss_smpc, sid_res['ssm'], open_loop_pred_samples_ss)

    # %% [markdown]
    """
    ## Closed-loop simulation
    """
    closed_loop_ss = sample_closed_loop(ss_smpc, sid_res, n_samples = 10, N_horizon=50)


    # %%

    fig, ax = plot_closed_loop_trajectory(closed_loop_ss)
    fig, ax = plot_closed_loop_cons_detail(closed_loop_ss, ss_smpc)
    # %%

    # %% [markdown]
    """
    ## Comparison
    """
    # %%
    ms_smpc = setup_controller(sid_res['msm'], smpc_settings)
    ss_smpc = setup_controller(sid_res['ssm'], smpc_settings)

    test_sys = get_system_for_case_study(sid_res['sigma_x'], sid_res['sigma_y'])

    ss_smpc.read_from(test_sys)
    ms_smpc.read_from(test_sys)

    ss_smpc(None, None)
    ms_smpc(None, None)

    # %%

    fig, ax = plt.subplots(3,2, figsize=(config_mpl.textwidth, .5*config_mpl.textwidth), 
        sharex=True, sharey='row', dpi=150,
        gridspec_kw = {'width_ratios':[1, 1], 'height_ratios':[2, 1, 1]}
        )

    _,_, ms_lines_open_loop = plot_open_loop_prediction_with_samples(
        ms_smpc, sid_res['msm'], open_loop_pred_samples_ms, fig_ax=(fig, ax[:,0]), with_annotations=False,
        )
    _,_, ss_lines_open_loop = plot_open_loop_prediction_with_samples(
        ss_smpc, sid_res['ssm'], open_loop_pred_samples_ss, fig_ax=(fig, ax[:,1]), with_annotations=False,
        )

    ax[0,0].set_title('SMPC w. multi-step model') 
    ax[0,1].set_title('SMPC w. state-space model') 
    ax[0,0].set_ylabel('room\n temp. [°C]')
    ax[1,0].set_ylabel('heat./ cool.\n power [kW]')
    ax[2,0].set_ylabel('amb.\n temp. [°C]')
    ax[2,0].set_xlabel('time [h]')
    ax[2,1].set_xlabel('time [h]')
    ax[2,0].text(-2.5, 7.5, '$\\leftarrow$ past')
    ax[2,0].text(.5, 7.5, 'pred. $\\rightarrow$')


    fig.align_ylabels()
    fig.tight_layout(pad=.1)
    dummy_lines = [
        ax[0,0].plot([], [], color='k', linestyle='-', label='open-loop pred.')[0],
        ax[0,0].plot([], [], color='k', linestyle='--', label=r'$\pm c_p\sigma$')[0],
        ax[0,0].plot([], [], color='k', linestyle='none', marker='$\equiv$', markersize=8, mew=.1, label='sampled closed-loop traj.', alpha=.4)[0],
        ax[0,0].plot([], [], color='k', linestyle=':', label='constraints')[0],
    ]

    ax[0,1].legend(handles=dummy_lines, loc='upper center', bbox_to_anchor=(0, 1.0), ncol=4, fontsize='small')

    dummy_lines = [
        ax[0,0].plot([], [], color=config_mpl.colors[0], linestyle='none', marker='s', label='1')[0],
        ax[0,0].plot([], [], color=config_mpl.colors[1], linestyle='none', marker='s', label='2')[0],
        ax[0,0].plot([], [], color=config_mpl.colors[2], linestyle='none', marker='s', label='3')[0],
        ax[0,0].plot([], [], color=config_mpl.colors[3], linestyle='none', marker='s', label='4')[0],
    ]
    ax[1,0].legend(handles=dummy_lines, loc='upper left', bbox_to_anchor=(0, 1.5), fontsize='small', title='room')

    savepath = os.path.join('..', '..', '2023_CDC_L-CSS_Paper_Stochastic_MSM', 'figures')
    savename = 'open_loop_pred_ms_vs_ss_smpc'
    fig.savefig(os.path.join(savepath, savename + '.pgf'), bbox_inches='tight', format='pgf')


# %%
