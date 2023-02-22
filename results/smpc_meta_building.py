# %% [markdown]
"""
# Meta analyis of stochastic MPC performance for the building model

## Import packages
"""
# %%
import numpy as np
import scipy
import pandas as pd
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
import copy

sys.path.append(os.path.join('..'))
sys.path.append(os.path.join('..', 'plots'))

import smpc
import sysid as sid
import helper
import system
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
        init_from_reference_sys: Optional[system.System] = None,
        ) -> system.System:
    
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

    # Generate an initial sequence of measurements or use the reference system as initial state
    if init_from_reference_sys is None:
        u_const = np.array([0,0,0,0,10]).reshape(-1,1) 
        random_input = sid.RandomInput(
            n_u=5, 
            u_lb = u_const,
            u_ub = u_const,
            u0= u_const,
            )

        # Generate an initial sequence of measurements
        sys.simulate(random_input, 3)
    else:
        for attr in ['_x', '_u', '_d', '_y', 't_now', '_time', 'x0']:
            setattr(sys, attr, copy.copy(getattr(init_from_reference_sys, attr)))


    if link_controller is not None:
        link_controller.read_from(sys)

    return sys

def sample_open_loop_prediction(
        controller: Union[smpc.StateSpaceSMPC, smpc.MultiStepSMPC],
        sid_res,
        reference_sys: system.System,
        n_samples: int = 50,
    ) -> sid.DataGenerator :

    open_loop_pred_input = sid.InputFromSequence(controller.res_u_pred)

    data_gen_setup = sid.DataGeneratorSetup(
        N=sid_res['msm'].data_setup.N,
        T_ini=0,
        n_samples = n_samples,
    )
    
    np.random.seed(99)

    data_gen = sid.DataGenerator(
        get_sys = functools.partial(get_system_for_case_study, 
                                    sid_res['sigma_x'], sid_res['sigma_y'],
                                    init_from_reference_sys=reference_sys),
        setup = data_gen_setup,
        u_fun = open_loop_pred_input,
    )

    return data_gen

def sample_closed_loop(
        controller: Union[smpc.StateSpaceSMPC, smpc.MultiStepSMPC],
        sid_res,
        reference_sys: system.System,
        n_samples: int = 10,
        N_horizon: int = 50,
    ) -> sid.DataGenerator :

    data_gen_setup = sid.DataGeneratorSetup(
        N=N_horizon,
        T_ini=0,
        n_samples = n_samples,
    )

    np.random.seed(99)

    data_gen = sid.DataGenerator(
        get_sys = functools.partial(
            get_system_for_case_study, 
            sid_res['sigma_x'], sid_res['sigma_y'], 
            link_controller=controller,
            init_from_reference_sys=reference_sys),
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
    lines['y_past'] = ax[0].plot(t_past, controller.res_y_past, '-')
    ax[0].axhline(18, color='k', linestyle=':')
    ax[0].axvline(0, color='k', linestyle='-')

    lines['u_pred'] = ax[1].step(t_pred[:-1],controller.res_u_pred[:-1,:4], where='post')
    ax[1].set_prop_cycle(None)
    lines['u_past'] = ax[1].step(t_past,controller.res_u_past[:,:4], '-', where='post')
    ax[1].axvline(0, color='k', linestyle='-')

    lines['t0_pred'] = ax[2].plot(t_pred[:-1],controller.res_u_pred[:-1,4])
    ax[2].set_prop_cycle(None)
    lines['t0_past'] = ax[2].plot(t_past,controller.res_u_past[:,4], '-')
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
        lines['t0_samples'].append(ax[2].plot(sim_res_k.time, sim_res_k.x[:,4], alpha=.2))

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
        )-> Tuple[plt.figure, plt.axis, Dict[str, plt.Line2D]]:
    
    if fig_ax is None:
        fig, ax = plt.subplots(1,1, figsize=(config_mpl.columnwidth, config_mpl.columnwidth))
    else:
        fig, ax = fig_ax


    lines = {}
    lines['y_samples'] = []
    for sim_res_k in closed_loop_res.sim_results:
        lines['y_samples'].append(ax.plot(sim_res_k.y[:,1], sim_res_k.y[:,0],color=config_mpl.colors[0], alpha=1/len(closed_loop_res.sim_results))[0])
    ax.plot([], [], color=config_mpl.colors[0], label=r'$r_k$', alpha=1)
    lines['y_pred'] = ax.plot(controller.res_y_pred[:,1], controller.res_y_pred[:,0], color=config_mpl.colors[1], label=r'$r_{k,\text{pred}}$')

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
    # ax.set_ylim(19, 21)
    # ax.set_xlim(17.5, 19.5)

    lines['cov0'] = e1
    lines['covN'] = e2

    if with_annotations:
        e1.set_label('covariance at t=0')
        e2.set_label('covariance at t=N')
        ax.set_xlabel('T2 [°C]')
        ax.set_ylabel('T1 [°C]')
        ax.legend()

    return fig, ax, lines

# %% [markdown]
"""
## Helper functions for KPIs
"""
def get_closed_loop_kpis(
        closed_loop_res: sid.DataGenerator,
        controller: Union[smpc.StateSpaceSMPC, smpc.MultiStepSMPC],
    ) -> pd.DataFrame:

    df_dict = {
        'sum_of_control_action': [],
        'cons_viol_perc': [],
    }

    for sim_res_k in closed_loop_res.sim_results:
        sum_of_control_action = np.sum(np.abs(sim_res_k.u[:,:4]))
        cons_viol_array = (controller.stage_cons_fun(sim_res_k.y.T) > controller._stage_cons.cons_ub).full()
        cons_viol_frac = np.sum(cons_viol_array) / cons_viol_array.size
        cons_viol_perc = np.round(cons_viol_frac * 100, 2)

        df_dict['sum_of_control_action'].append(sum_of_control_action)
        df_dict['cons_viol_perc'].append(cons_viol_perc)

    df = pd.DataFrame(df_dict)

    return df

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

    # Reference system is simulated with random initial sequence.
    # This random initial sequence can be transfereed to the test system
    ref_sys = get_system_for_case_study(sid_res['sigma_x'], sid_res['sigma_y'])    

    test_sys = get_system_for_case_study(sid_res['sigma_x'], sid_res['sigma_y'], init_from_reference_sys=ref_sys)

    # %%
    print(ms_smpc.cp)


    # %%

    ms_smpc.read_from(test_sys)
    # Simulate the system using the controller for n time steps
    ms_smpc(None, None)

    _ = plot_open_loop_prediction(ms_smpc, sid_res['msm'])

    # %% [markdown]
    """
    Sample the real system with this open-loop prediction
    """
    open_loop_pred_samples_ms = sample_open_loop_prediction(ms_smpc, sid_res, n_samples = 50, reference_sys=ref_sys)

    _ = plot_open_loop_prediction_with_samples(ms_smpc, sid_res['msm'], open_loop_pred_samples_ms)


    # %% [markdown]
    """
    ## Closed-loop simulation

    Sample and save results. Alternatively, load the saved results.
    """
    savepath = os.path.join('smpc_results')
    savename = '02_ms_smpc_closed_loop_results_with_cov.pkl'

    if False:
        print('Sampling closed-loop results... (this may take a while)')
        closed_loop_ms = sample_closed_loop(ms_smpc, sid_res, n_samples = 10, N_horizon=50, reference_sys=ref_sys)

        with open(os.path.join(savepath, savename), 'wb') as f:
            pickle.dump(closed_loop_ms, f)
    elif os.path.exists(os.path.join(savepath, savename)):
        print('Loading closed-loop results from file... make sure no settings have changed!')
        with open(os.path.join(savepath, savename), 'rb') as f:
            closed_loop_ms = pickle.load(f)
    else:
        print('No closed-loop results found. Please run the sampling first.')
    
    # %%

    _ = plot_closed_loop_trajectory(closed_loop_ms)
    _ = plot_closed_loop_cons_detail(closed_loop_ms, ms_smpc)

    get_closed_loop_kpis(closed_loop_ms, ms_smpc).mean()
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

    test_sys = get_system_for_case_study(sid_res['sigma_x'], sid_res['sigma_y'], init_from_reference_sys=ref_sys)

    # %%
    ss_smpc.read_from(test_sys)
    # Simulate the system using the controller for n time steps
    ss_smpc(None, None)

    _ = plot_open_loop_prediction(ss_smpc, sid_res['ssm'])


    # %% [markdown]
    """
    Sample the real system with this open-loop prediction
    """
    _ = open_loop_pred_samples_ss = sample_open_loop_prediction(ss_smpc, sid_res, n_samples = 50, reference_sys=ref_sys)

    _ = plot_open_loop_prediction_with_samples(ss_smpc, sid_res['ssm'], open_loop_pred_samples_ss)

    # %% [markdown]
    """
    ## Closed-loop simulation
    """
    savepath = os.path.join('smpc_results')
    savename = '02_ss_smpc_closed_loop_results_with_cov.pkl'

    if False:
        print('Sampling closed-loop results... (this may take a while)')
        closed_loop_ss = sample_closed_loop(ss_smpc, sid_res, n_samples = 10, N_horizon=50, reference_sys=ref_sys)

        with open(os.path.join(savepath, savename), 'wb') as f:
            pickle.dump(closed_loop_ss, f)
    elif os.path.exists(os.path.join(savepath, savename)):
        print('Loading closed-loop results from file... make sure no settings have changed!')
        with open(os.path.join(savepath, savename), 'rb') as f:
            closed_loop_ss = pickle.load(f)
    else:
        print('No closed-loop results found. Please run the sampling first.')
    # %%

    _ = plot_closed_loop_trajectory(closed_loop_ss)
    _ = plot_closed_loop_cons_detail(closed_loop_ss, ss_smpc)
    # %%

    # %% [markdown]
    """
    ## Comparison
    """
    # %%
    ms_smpc = setup_controller(sid_res['msm'], smpc_settings)
    ss_smpc = setup_controller(sid_res['ssm'], smpc_settings)

    test_sys = get_system_for_case_study(sid_res['sigma_x'], sid_res['sigma_y'], init_from_reference_sys=ref_sys)

    ss_smpc.read_from(test_sys)
    ms_smpc.read_from(test_sys)

    ss_smpc(None, None)
    ms_smpc(None, None)

    _ = open_loop_pred_samples_ss = sample_open_loop_prediction(ss_smpc, sid_res, n_samples = 50, reference_sys=ref_sys)
    _ = open_loop_pred_samples_ms = sample_open_loop_prediction(ms_smpc, sid_res, n_samples = 50, reference_sys=ref_sys)

    # %%

    fig, ax = plt.subplots(3,2, figsize=(config_mpl.textwidth, .4*config_mpl.textwidth), 
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

    savepath = os.path.join('smpc_results')
    savename_ss = '02_{}_smpc_closed_loop_results_with_cov.pkl'

    with open(os.path.join(savepath, savename_ss.format('ss')), 'rb') as f:
        closed_loop_ss = pickle.load(f)
    with open(os.path.join(savepath, savename_ss.format('ms')), 'rb') as f:
        closed_loop_ms = pickle.load(f)

    # %%

    fig, ax = plt.subplots(3,2, figsize=(config_mpl.textwidth, .4*config_mpl.textwidth), 
        sharex=True, sharey='row', dpi=150,
        gridspec_kw = {'width_ratios':[1, 1], 'height_ratios':[2, 1, 1]}
        )
    
    _, _, ms_lines_closed_loop = plot_closed_loop_trajectory(closed_loop_ms, fig_ax=(fig, ax[:,0]), with_annotations=False)
    _, _, ss_lines_closed_loop = plot_closed_loop_trajectory(closed_loop_ss, fig_ax=(fig, ax[:,1]), with_annotations=False)
    

    ax[0,0].set_title('SMPC w. multi-step model') 
    ax[0,1].set_title('SMPC w. state-space model') 
    ax[0,0].set_ylabel('room\n temp. [°C]')
    ax[1,0].set_ylabel('heat./ cool.\n power [kW]')
    ax[2,0].set_ylabel('amb.\n temp. [°C]')
    ax[2,0].set_xlabel('time [h]')
    ax[2,1].set_xlabel('time [h]')

    fig.align_ylabels()
    fig.tight_layout(pad=.1)

    dummy_lines = [
        ax[0,0].plot([], [], color=config_mpl.colors[0], linestyle='none', marker='s', label='1')[0],
        ax[0,0].plot([], [], color=config_mpl.colors[1], linestyle='none', marker='s', label='2')[0],
        ax[0,0].plot([], [], color=config_mpl.colors[2], linestyle='none', marker='s', label='3')[0],
        ax[0,0].plot([], [], color=config_mpl.colors[3], linestyle='none', marker='s', label='4')[0],
    ]
    ax[0,0].legend(handles=dummy_lines, loc='upper right', bbox_to_anchor=(1, 1.1), fontsize='small', title='room')

    savepath = os.path.join('..', '..', '2023_CDC_L-CSS_Paper_Stochastic_MSM', 'figures')
    savename = 'closed_loop_pred_ms_vs_ss_smpc'
    fig.savefig(os.path.join(savepath, savename + '.pgf'), bbox_inches='tight', format='pgf')


    # %% [markdown]
    """
    ## Comparison of SMPC with and without full covariance estimation
    """
    # %%
    smpc_settings_wo_cov = smpc.base.SMPCSettings(
        prob_chance_cons=smpc_settings.prob_chance_cons,
        with_cov=False
    )
    smpc_settings_wo_cov.surpress_ipopt_output()
    ms_smpc_with_cov = setup_controller(sid_res['msm'], smpc_settings)
    ms_smpc_wo_cov = setup_controller(sid_res['msm'], smpc_settings_wo_cov)

    closed_loop_ms_with_cov = sample_closed_loop(ms_smpc_with_cov, sid_res, n_samples = 1, N_horizon=50, reference_sys=ref_sys)
    closed_loop_ms_wo_cov = sample_closed_loop(ms_smpc_wo_cov, sid_res, n_samples = 1, N_horizon=50, reference_sys=ref_sys)


    # %%

    fig, ax = plt.subplots(2,1, figsize=(config_mpl.columnwidth, config_mpl.columnwidth), sharex=True, dpi=150)

    _,_,lines_ms = plot_closed_loop_cons_detail(closed_loop_ms_with_cov, ms_smpc_with_cov, fig_ax=(fig, ax[0]), with_annotations=False)
    _,_,lines_ss = plot_closed_loop_cons_detail(closed_loop_ms_wo_cov, ms_smpc_wo_cov, fig_ax=(fig, ax[1]), with_annotations=False)
    ax[0].axis('equal')
    ax[1].axis('equal')
    ax[0].set_ylim(18, 21)
    ax[1].set_ylim(18.5, 21.5)
    ax[0].set_xlim(17.5, 20.5)

    ax[0].set_title('MSM-SMPC w. full covariance estimation')
    ax[1].set_title('MSM-SMPC w. only variance estimation')

    ax[0].set_ylabel('temp. room $t_1$ [°C]')
    ax[1].set_ylabel('temp. room $t_1$ [°C]')
    ax[1].set_xlabel('temp. room $t_2$ [°C]')

    lines_ms['cov0'].set_label('est. cov. at $k+1$')
    lines_ms['covN'].set_label('est. cov. at $k+N$')

    ax[1].annotate(r'$t_1 - t_2 \geq 1$', xy=(19, 20), xytext=(19.5, 19.5), arrowprops=dict(facecolor='black', shrink=0.05, width=.1, headwidth=4, headlength=3))

    ax[0].legend(loc='lower right')

    fig.tight_layout(pad=.2)

    savepath = os.path.join('..', '..', '2023_CDC_L-CSS_Paper_Stochastic_MSM', 'figures')
    savename = 'closed_loop_detail_ms_smpc_with_vs_wo_cov'
    fig.savefig(os.path.join(savepath, savename + '.pgf'), bbox_inches='tight', format='pgf')
    # %% [markdown]
    """
    ## Full sampling of the controller (SS and MS) with only variance information for KPI evaluation
    """

    ms_smpc_wo_cov = setup_controller(sid_res['msm'], smpc_settings_wo_cov)
    ss_smpc_wo_cov = setup_controller(sid_res['ssm'], smpc_settings_wo_cov)

    savepath = os.path.join('smpc_results')
    savename = '02_ss_smpc_closed_loop_results_wo_cov.pkl'

    if True:
        print('Sampling closed-loop results... (this may take a while)')
        closed_loop_ss_wo_cov = sample_closed_loop(ss_smpc_wo_cov, sid_res, n_samples = 10, N_horizon=50, reference_sys=ref_sys)

        with open(os.path.join(savepath, savename), 'wb') as f:
            pickle.dump(closed_loop_ss_wo_cov, f)
    elif os.path.exists(os.path.join(savepath, savename)):
        print(f'Loading closed-loop results from file {savename}... make sure no settings have changed!')
        with open(os.path.join(savepath, savename), 'rb') as f:
            closed_loop_ss_wo_cov = pickle.load(f)
    else:
        print('No closed-loop results found. Please run the sampling first.')

    savename = '02_ms_smpc_closed_loop_results_wo_cov.pkl'

    if True:
        print('Sampling closed-loop results... (this may take a while)')
        closed_loop_ms_wo_cov = sample_closed_loop(ms_smpc_wo_cov, sid_res, n_samples = 10, N_horizon=50, reference_sys=ref_sys)

        with open(os.path.join(savepath, savename), 'wb') as f:
            pickle.dump(closed_loop_ms_wo_cov, f)
    elif os.path.exists(os.path.join(savepath, savename)):
        print(f'Loading closed-loop results from file {savename}... make sure no settings have changed!')
        with open(os.path.join(savepath, savename), 'rb') as f:
            closed_loop_ms_wo_cov = pickle.load(f)
    else:
        print('No closed-loop results found. Please run the sampling first.')


    # %%
    fig, ax = plt.subplots(3,2, figsize=(config_mpl.textwidth, .4*config_mpl.textwidth), 
        sharex=True, sharey='row', dpi=150,
        gridspec_kw = {'width_ratios':[1, 1], 'height_ratios':[2, 1, 1]}
        )
    
    _, _, ms_lines_closed_loop = plot_closed_loop_trajectory(closed_loop_ms_wo_cov, fig_ax=(fig, ax[:,0]), with_annotations=False)
    _, _, ss_lines_closed_loop = plot_closed_loop_trajectory(closed_loop_ss_wo_cov, fig_ax=(fig, ax[:,1]), with_annotations=False)
    

    # %% [markdown]
    """
    ## Read KPI and export to table
    """
    # %%
    kpi_closed_loop_ms_with_cov = get_closed_loop_kpis(closed_loop_ms, ms_smpc)
    kpi_closed_loop_ss_with_cov = get_closed_loop_kpis(closed_loop_ss, ss_smpc)
    kpi_closed_loop_ms_wo_cov = get_closed_loop_kpis(closed_loop_ms_wo_cov, ms_smpc)
    kpi_closed_loop_ss_wo_cov = get_closed_loop_kpis(closed_loop_ss_wo_cov, ss_smpc)

    kpi_closed_loop_ss_wo_cov

    df_ms = pd.concat([
        kpi_closed_loop_ms_with_cov.mean(),
        kpi_closed_loop_ms_wo_cov.mean(),
    ], axis=1, keys=['with cov.', 'w/o cov.'])

    df_ss = pd.concat([
        kpi_closed_loop_ss_with_cov.mean(),
        kpi_closed_loop_ss_wo_cov.mean(),
    ], axis=1, keys=['with cov.', 'w/o cov.'])

    df_cat = pd.concat([df_ms, df_ss], axis=1, keys=['MSM-SMPC', 'SSM-SMPC'])

    df_cat

    # %%


    tex_str = df_cat.to_latex(
        float_format='{:0.2f}'.format,
        multicolumn=True,
        multirow=True,
        # column_format='p{2.5cm}'+'X'*(len(df.columns)),
    )

    tex_str = tex_str.replace('sum\\_of\\_control\\_action', r'$\bar{q}$ [kWh]')
    tex_str = tex_str.replace('cons\\_viol\\_perc', r'cons. viol. [\%]')


    tex_str_list = tex_str.split('\n')

    tex_str_list.insert(3, r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}')

    tex_str_list.pop(1) # Remove toprule


    tex_str =  '\n'.join(tex_str_list)

    
    savepath = os.path.join('..', '..', '2023_CDC_L-CSS_Paper_Stochastic_MSM', 'tables')
    savename = 'closed_loop_results_table.tex'

    with open(os.path.join(savepath, savename), 'w') as f:
        f.write(tex_str)



# %%
