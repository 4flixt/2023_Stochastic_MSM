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
import matplotlib as mpl
import importlib
import functools
import time
import random
import copy


sys.path.append(os.path.join('..', '..'))
from blrsmpc.sysid import sysid as sid
from blrsmpc import smpc
from blrsmpc import system
from blrsmpc import helper
from blrsmpc import plotconfig


# Configure matplotlib and load notation from tex
plotconfig.config_mpl(os.path.join('..', '..', 'blrsmpc', 'plotconfig', 'notation.tex'))
colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

# Export figures?
export_figures = True

# %% [markdown]
"""
## Helper functions for controller setup
"""

# %%
def load_sid_results(file_name: str):
    load_name = os.path.join('sid_results', file_name)
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
    )

    y = controller._y_stage
    # controller.set_chance_cons(expr =  y, ub = 30)
    controller.set_chance_cons(expr = -y[:4], ub = -18)
    # y[0]-y[1] >= 1
    # controller.set_chance_cons(expr =  -y[0]+y[1], ub = -1)

    controller.setup()

    # Set bounds
    controller.lb_opt_x['u_pred',:] = np.array([-6, -6, -6, -6, 10])
    controller.ub_opt_x['u_pred',:] = np.array([6, 6, 6, 6, 10])

    return controller

def get_system_for_case_study(
        sig_x, 
        sig_y, 
        T_ini: int,
        t_0_rooms = np.array([23,20,20,20]), 
        t_ambient = 10, 
        link_controller: Optional[Union[smpc.StateSpaceSMPC, smpc.MultiStepSMPC]] = None,
        init_from_reference_sys: Optional[system.System] = None,
        ) -> system.System:
    
    x0 = np.concatenate((
        np.atleast_2d(t_0_rooms).reshape(-1,1),
        np.atleast_2d(t_ambient).reshape(-1,1),
    ), axis=0)

    if sig_y.shape[0] == 4:
        state_feedback = False
    elif sig_y.shape[0] == 5:
        state_feedback = True

    sys_generator = sid.SystemGenerator(
            sys_type=sid.SystemType.BUILDING,
            sig_x=sig_x,
            sig_y=sig_y,
            case_kwargs={'state_feedback':state_feedback, 'x0': x0},
            dt=3600,
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
        sys.simulate(random_input, )
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
                                    sid_res['sigma_x'], sid_res['sigma_y'], sid_res['t_ini'],
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
        dt = 3600
    )

    np.random.seed(99)

    data_gen = sid.DataGenerator(
        get_sys = functools.partial(
            get_system_for_case_study, 
            sid_res['sigma_x'], sid_res['sigma_y'], sid_res['t_ini'],
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

    print(f'Sigma_y shape = {sid_model.data_setup.sig_y.shape[0]}')

    if sid_model.data_setup.sig_y.shape[0] == 5:
        state_feedback = True
    elif sid_model.data_setup.sig_y.shape[0] == 4:
        state_feedback = False
    else:
        raise ValueError('sig_y has wrong shape')

    lines['y_pred'] = ax[0].plot(t_pred, controller.res_y_pred[:,:4])
    ax[0].set_prop_cycle(None)
    ax[0].plot(t_pred, controller.res_y_pred[:,:4]+controller.cp*controller.res_y_std[:,:4], '--')
    ax[0].set_prop_cycle(None)
    ax[0].plot(t_pred, controller.res_y_pred[:,:4]-controller.cp*controller.res_y_std[:,:4], '--')
        
    ax[0].set_prop_cycle(None)
    lines['y_past'] = ax[0].plot(t_past, controller.res_y_past[:,:4], 'x')
    ax[0].axhline(18, color='k', linestyle=':')
    ax[0].axvline(0, color='k', linestyle='-')

    lines['u_pred'] = ax[1].step(t_pred[:-1],controller.res_u_pred[:-1,:4], where='post')
    ax[1].set_prop_cycle(None)
    lines['u_past'] = ax[1].step(t_past,controller.res_u_past[:,:4], 'x', where='post')
    ax[1].axvline(0, color='k', linestyle='-')

    # lines['t0_pred'] = ax[2].plot(t_pred[:-1],controller.res_u_pred[:-1,4])

    if state_feedback:
        lines['y_pred_t0'] = ax[2].plot(t_pred, controller.res_y_pred[:,4], color=colors[0])
        ax[2].plot(t_pred, controller.res_y_pred[:,4]+controller.cp*controller.res_y_std[:,4], '--', color=colors[0])
        ax[2].plot(t_pred, controller.res_y_pred[:,4]-controller.cp*controller.res_y_std[:,4], '--', color=colors[0])

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
    with_intial_sequence: bool = False,
    ) -> Tuple[plt.figure, plt.axis, Dict[str, plt.Line2D]]:


    fig, ax, lines = plot_open_loop_prediction(controller, sid_model, fig_ax, with_annotations)

    lines['y_samples'] = []
    lines['t0_samples'] = []

    if with_intial_sequence:
        start_ind = 0
    else:
        start_ind = sid_model.data_setup.T_ini


    dt = sid_model.data_setup.dt



    for sim_res in open_loop_samples.sim_results:
        time = sim_res.time[start_ind:]/dt - sim_res.time[start_ind]/dt
        ax[0].set_prop_cycle(None)
        lines['y_samples'].append(ax[0].plot(time, sim_res.y[start_ind:,:4], alpha=.1))

        ax[-1].set_prop_cycle(None)
        lines['t0_samples'].append(ax[-1].plot(time, sim_res.x[start_ind:,4], alpha=.1))

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

    dt = closed_loop_res.setup.dt

    for sim_res_k in closed_loop_res.sim_results:
        lines['y_samples'].append(ax[0].plot(sim_res_k.time/dt, sim_res_k.y[:,:4], alpha=.2))
        lines['u_samples'].append(ax[1].step(sim_res_k.time/dt, sim_res_k.u[:,:4], where='post', alpha=.2))
        lines['t0_samples'].append(ax[2].plot(sim_res_k.time/dt, sim_res_k.x[:,4], alpha=.2))

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
        fig, ax = plt.subplots(1,1, figsize=(plotconfig.columnwidth, plotconfig.columnwidth))
    else:
        fig, ax = fig_ax


    lines = {}
    lines['y_samples'] = []
    for sim_res_k in closed_loop_res.sim_results:
        lines['y_samples'].append(ax.plot(sim_res_k.y[:,1], sim_res_k.y[:,0],color=plotconfig.colors[0], alpha=1/len(closed_loop_res.sim_results))[0])
    ax.plot([], [], color=plotconfig.colors[0], label=r'$\vy_{[0,k]}$', alpha=1)
    lines['y_pred'] = ax.plot(controller.res_y_pred[:,1], controller.res_y_pred[:,0], color=plotconfig.colors[1], label=r'$\hat\vy_{[k+1,k+N]}$')

    cons_1 = np.linspace(17.8, 20, 5)
    ax.plot(cons_1, cons_1+1, color='k', linestyle='--', label='constraint')
    ax.axvline(18, color='k', linestyle='--')

    cov0 = controller.opt_aux_num['Sigma_y_pred'][:2,:2].full()
    covN = controller.opt_aux_num['Sigma_y_pred'][-4:-2,-4:-2].full()

    e2=helper.plot_cov_as_ellipse(controller.res_y_pred[-1,1], controller.res_y_pred[-1,0], covN,
        ax=ax, n_std=controller.cp, edgecolor=plotconfig.colors[2], facecolor=plotconfig.colors[2], alpha=0.3
        )
    e1=helper.plot_cov_as_ellipse(controller.res_y_pred[0,1], controller.res_y_pred[0,0], cov0,
        ax=ax, n_std=controller.cp, edgecolor=plotconfig.colors[1], facecolor=plotconfig.colors[1], alpha=0.3
        )
    # ax.set_ylim(19, 21)
    # ax.set_xlim(17.5, 19.5)

    lines['cov0'] = e1
    lines['covN'] = e2

    if with_annotations:
        e1.set_label('$p(T_1,T_2)$ at $t=k$')
        e2.set_label('$p(T_1,T_2)$ at $t=k+N$')
        ax.set_xlabel('$T_2$ [$^\circ$C]')
        ax.set_ylabel('$T_1$ [$^\circ$C]')
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
    np.random.seed(99)

    sid_result_file_name = 'sid_building_tini_1_n_12.pkl'

    sid_res = load_sid_results(sid_result_file_name)

    smpc_settings = smpc.base.SMPCSettings(
        prob_chance_cons=.999,
        with_cov=True,
    )
    smpc_settings.surpress_ipopt_output()

    ms_smpc = setup_controller(sid_res['msm'], smpc_settings)

    # Reference system is simulated with random initial sequence.
    # This random initial sequence can be transfereed to the test system
    ref_sys = get_system_for_case_study(sid_res['sigma_x'], sid_res['sigma_y'], sid_res['t_ini'])    

    test_sys = get_system_for_case_study(sid_res['sigma_x'], sid_res['sigma_y'], sid_res['t_ini'], init_from_reference_sys=ref_sys)

    # %%
    np.random.seed(99)

    ms_smpc.read_from(test_sys)
    # Simulate the system using the controller for n time steps
    ms_smpc(None, None)

    _ = plot_open_loop_prediction(ms_smpc, sid_res['msm'])

    # %% [markdown]
    """
    Sample the real system with this open-loop prediction
    """
    open_loop_pred_samples_ms = sample_open_loop_prediction(ms_smpc, sid_res, n_samples = 50, reference_sys=ref_sys)

    fig, ax = plt.subplots(3,1, gridspec_kw={'height_ratios': [2,1,1]}, sharex=True)
    _ = plot_open_loop_prediction_with_samples(ms_smpc, sid_res['msm'], open_loop_pred_samples_ms, fig_ax=(fig, ax))


    # %% [markdown]
    """
    ## Closed-loop simulation

    Sample and save results. Alternatively, load the saved results.
    """
    np.random.seed(99)

    savepath = os.path.join('smpc_results')
    savename = '06_ms_smpc_closed_loop_results_with_cov_20_state_fb.pkl'
    overwrite = False
        
    if os.path.exists(os.path.join(savepath, savename)) and not overwrite:
        print('Loading closed-loop results from file... make sure no settings have changed!')
        with open(os.path.join(savepath, savename), 'rb') as f:
            closed_loop_ms = pickle.load(f)
    else:
        print('Sampling closed-loop results... (this may take a while)')
        closed_loop_ms = sample_closed_loop(ms_smpc, sid_res, n_samples = 20, N_horizon=50, reference_sys=ref_sys)

        with open(os.path.join(savepath, savename), 'wb') as f:
            pickle.dump(closed_loop_ms, f)
    
    # %%

    _ = plot_closed_loop_trajectory(closed_loop_ms)
    # _ = plot_closed_loop_cons_detail(closed_loop_ms, ms_smpc)
    get_closed_loop_kpis(closed_loop_ms, ms_smpc).mean()
    # %%

    # %% [markdown]
    """
    ### SMPC with state-space model
    """
    # %%

    # sid_res = load_sid_results(sid_result_file_name)

    smpc_settings = smpc.base.SMPCSettings(
        prob_chance_cons=.999,
        with_cov=True,
    )
    smpc_settings.surpress_ipopt_output()
    smpc_settings.nlp_opts['ipopt.max_iter'] = 500

    ss_smpc = setup_controller(sid_res['ssm'], smpc_settings)

    test_sys = get_system_for_case_study(sid_res['sigma_x'], sid_res['sigma_y'], sid_res['t_ini'], init_from_reference_sys=ref_sys)

    # %%
    ss_smpc.read_from(test_sys)
    # Simulate the system using the controller for n time steps
    ss_smpc(None, None)

    _ = plot_open_loop_prediction(ss_smpc, sid_res['ssm'])


    # %% [markdown]
    """
    Sample the real system with this open-loop prediction
    """
    np.random.seed(99)
    _ = open_loop_pred_samples_ss = sample_open_loop_prediction(ss_smpc, sid_res, n_samples = 50, reference_sys=ref_sys)

    _ = plot_open_loop_prediction_with_samples(ss_smpc, sid_res['ssm'], open_loop_pred_samples_ss)

    # %% [markdown]
    """
    ## Closed-loop simulation
    """
    np.random.seed(99)

    savepath = os.path.join('smpc_results')
    savename = '06_ss_smpc_closed_loop_results_with_cov_20_state_fb.pkl'
    overwrite = False

    if os.path.exists(os.path.join(savepath, savename)) and not overwrite:
        print('Loading closed-loop results from file... make sure no settings have changed!')
        with open(os.path.join(savepath, savename), 'rb') as f:
            closed_loop_ss = pickle.load(f)
    else:
        print('Sampling closed-loop results... (this may take a while)')
        closed_loop_ss = sample_closed_loop(ss_smpc, sid_res, n_samples = 20, N_horizon=50, reference_sys=ref_sys)

        with open(os.path.join(savepath, savename), 'wb') as f:
            pickle.dump(closed_loop_ss, f)
    # %%

    _ = plot_closed_loop_trajectory(closed_loop_ss)
    _ = plot_closed_loop_cons_detail(closed_loop_ss, ss_smpc)
    get_closed_loop_kpis(closed_loop_ss, ss_smpc).mean()
    # %%

    # %% [markdown]
    """
    ## Comparison
    """
    # %%
    ms_smpc = setup_controller(sid_res['msm'], smpc_settings)
    ss_smpc = setup_controller(sid_res['ssm'], smpc_settings)

    test_sys = get_system_for_case_study(sid_res['sigma_x'], sid_res['sigma_y'], sid_res['t_ini'], init_from_reference_sys=ref_sys)

    ss_smpc.read_from(test_sys)
    ms_smpc.read_from(test_sys)

    # Call controller with reading from system (call with None, None)
    ss_smpc(None, None)
    ms_smpc(None, None)

    _ = open_loop_pred_samples_ss = sample_open_loop_prediction(ss_smpc, sid_res, n_samples = 50, reference_sys=ref_sys)
    _ = open_loop_pred_samples_ms = sample_open_loop_prediction(ms_smpc, sid_res, n_samples = 50, reference_sys=ref_sys)

    # %%

    fig, ax = plt.subplots(3,2, figsize=(plotconfig.textwidth, .5*plotconfig.textwidth), 
        sharex=True, sharey='row', dpi=150,
        gridspec_kw = {'width_ratios':[1, 1], 'height_ratios':[3, 1, 1]}
        )

    _,_, ms_lines_open_loop = plot_open_loop_prediction_with_samples(
        ms_smpc, sid_res['msm'], open_loop_pred_samples_ms, fig_ax=(fig, ax[:,0]), with_annotations=False,
        )
    _,_, ss_lines_open_loop = plot_open_loop_prediction_with_samples(
        ss_smpc, sid_res['ssm'], open_loop_pred_samples_ss, fig_ax=(fig, ax[:,1]), with_annotations=False,
        )

    ax[0,0].set_title('SMPC with multi-step model (MSM-SMPC)') 
    ax[0,1].set_title('SMPC with state-space model (SSM-SMPC)') 
    ax[0,0].set_ylabel('room\n temp. [°C]')
    ax[1,0].set_ylabel('heat./ cool.\n power [kW]')
    ax[2,0].set_ylabel('amb.\n temp. [°C]')
    ax[2,0].set_xlabel('time [h]')
    ax[2,1].set_xlabel('time [h]')
    # ax[2,0].text(-1.6, 8, '$\\leftarrow$ past')
    ax[2,0].text(.5, 8, 'pred. $\\rightarrow$')


    fig.align_ylabels()
    fig.tight_layout(pad=.1)
    dummy_lines = [
        ax[0,0].plot([], [], color='k', marker='x', markersize = 4, linestyle='none', label='initial measurement')[0],
        ax[0,0].plot([], [], color='k', linestyle='-', label='open-loop pred.')[0],
        ax[0,0].plot([], [], color='k', linestyle='--', label=r'$\pm c_p\sigma$')[0],
        ax[0,0].plot([], [], color='k', linestyle='none', marker='$\equiv$', markersize=8, mew=.1, label='sampled closed-loop traj.', alpha=.4)[0],
        ax[0,0].plot([], [], color='k', linestyle=':', label='constraints')[0],
    ]

    ax[0,1].legend(handles=dummy_lines, loc='upper center', bbox_to_anchor=(0, 1.0), ncol=5, fontsize='small')

    dummy_lines = [
        ax[0,0].plot([], [], color=plotconfig.colors[0], linestyle='none', marker='s', label='1')[0],
        ax[0,0].plot([], [], color=plotconfig.colors[1], linestyle='none', marker='s', label='2')[0],
        ax[0,0].plot([], [], color=plotconfig.colors[2], linestyle='none', marker='s', label='3')[0],
        ax[0,0].plot([], [], color=plotconfig.colors[3], linestyle='none', marker='s', label='4')[0],
    ]
    ax[0,0].legend(handles=dummy_lines, ncols=4, loc='upper center', bbox_to_anchor=(.5, .85), fontsize='small', title='room')

    savepath = os.path.join('..', '..', '..', '2023_CDC_L-CSS_Paper_Stochastic_MSM', 'figures')
    savename = 'open_loop_pred_ms_vs_ss_smpc'
    if export_figures:
        fig.savefig(os.path.join(savepath, savename + '.pgf'), bbox_inches='tight', format='pgf')

    # %%

    # %%

    fig, ax = plt.subplots(3,2, figsize=(plotconfig.textwidth, .4*plotconfig.textwidth), 
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
        ax[0,0].plot([], [], color=plotconfig.colors[0], linestyle='none', marker='s', label='1')[0],
        ax[0,0].plot([], [], color=plotconfig.colors[1], linestyle='none', marker='s', label='2')[0],
        ax[0,0].plot([], [], color=plotconfig.colors[2], linestyle='none', marker='s', label='3')[0],
        ax[0,0].plot([], [], color=plotconfig.colors[3], linestyle='none', marker='s', label='4')[0],
    ]
    ax[0,0].legend(handles=dummy_lines, loc='upper right', bbox_to_anchor=(1, 1.1), fontsize='small', title='room')

    savepath = os.path.join('..', '..', '..', '2023_CDC_L-CSS_Paper_Stochastic_MSM', 'figures')
    savename = 'closed_loop_pred_ms_vs_ss_smpc'
    if export_figures:
        fig.savefig(os.path.join(savepath, savename + '.pgf'), bbox_inches='tight', format='pgf')


    # %% [markdown]
    """
    ## Read KPI and export to table
    """
    # %%
    kpi_closed_loop_ms_with_cov = get_closed_loop_kpis(closed_loop_ms, ms_smpc)
    kpi_closed_loop_ss_with_cov = get_closed_loop_kpis(closed_loop_ss, ss_smpc)

    def mean_std(x):
        return '{:.1f}+-{:.1f}'.format(x.mean(),x.std())

    df_cat = pd.concat([
        kpi_closed_loop_ms_with_cov.apply(mean_std, axis=0),
        kpi_closed_loop_ss_with_cov.apply(mean_std, axis=0),
    ], axis=1, keys=['MSM-SMPC', 'SSM-SMPC'])


    df_cat



    # %%


    tex_str = df_cat.to_latex(
        float_format='{:0.2f}'.format,
        multicolumn=True,
        multirow=True,
        # column_format='p{2.5cm}'+'X'*(len(df.columns)),
    )

    tex_str = tex_str.replace(r'sum_of_control_action', r'$\sum_i Q_i$ [kWh]')
    tex_str = tex_str.replace(r'cons_viol_perc', r'cons. viol. [\%]')
    tex_str = tex_str.replace(r'+-', r'$\pm$')


    tex_str_list = tex_str.split('\n')

    # tex_str_list.insert(3, r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}')

    tex_str_list.pop(1) # Remove toprule


    tex_str =  '\n'.join(tex_str_list)


    
    savepath = os.path.join('..', '..', '..', '2023_CDC_L-CSS_Paper_Stochastic_MSM', 'tables')
    savename = 'closed_loop_results_table.tex'

    with open(os.path.join(savepath, savename), 'w') as f:
        f.write(tex_str)

    tex_str


# %%
