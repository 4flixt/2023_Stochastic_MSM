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

# Get colors
import matplotlib as mpl
colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

sys.path.append(os.path.join('..'))

import smpc
import sysid as sid
import helper
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

def get_system_for_case_study(sig_x, sig_y, t_0_rooms = np.array([24,20,20,20]), t_ambient = 10):
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

# %% [markdown]
"""
## Helper functions for plotting
"""
# %%

def plot_open_loop_prediction(
        controller: Union[smpc.StateSpaceSMPC, smpc.MultiStepSMPC],
        sid_model: Union[sid.StateSpaceModel, sid.MultistepModel],
        fig_ax: Optional[Tuple[plt.figure, plt.axis]] = None
        ) -> Tuple[plt.figure, plt.axis]:
    
    t_past = np.arange(-sid_model.data_setup.T_ini,0)
    t_pred = np.arange(sid_model.data_setup.N)

    if fig_ax is None:
        fig, ax = plt.subplots(3,1, sharex=True, figsize=(8, 10))
    else:
        fig, ax = fig_ax

    meas_lines = ax[0].plot(t_pred, controller.res_y_pred)

    ax[0].set_prop_cycle(None)
    if False:
        for k in range(sid_model.n_y):
            ax[0].fill_between(t_pred, 
                            controller.res_y_pred[:,k]+controller.cp*controller.res_y_std[:,k], 
                            controller.res_y_pred[:,k]-controller.cp*controller.res_y_std[:,k], 
                            alpha=.2)
    else:
        ax[0].plot(t_pred, controller.res_y_pred+controller.cp*controller.res_y_std, '--')
        ax[0].set_prop_cycle(None)
        ax[0].plot(t_pred, controller.res_y_pred-controller.cp*controller.res_y_std, '--')
        
    ax[0].set_prop_cycle(None)
    ax[0].plot(t_past, controller.res_y_past, '-x')
    ax[0].axhline(18, color='k', linestyle='--')
    ax[0].axvline(0, color='k', linestyle='-')
    ax[0].text(-2, 18.2, '$\\leftarrow$ past')
    ax[0].text(.5, 18.2, 'pred. $\\rightarrow$')

    ax[1].step(t_pred[:-1],controller.res_u_pred[:-1,:4], where='post')
    ax[1].set_prop_cycle(None)
    ax[1].step(t_past,controller.res_u_past[:,:4], '-x', where='post')

    ax[2].plot(t_pred[:-1],controller.res_u_pred[:-1,4])
    ax[2].set_prop_cycle(None)
    ax[2].plot(t_past,controller.res_u_past[:,4], '-x')

    ax[0].legend(meas_lines, ['room 1', 'room 2', 'room 3', 'room 4'], ncol=4)
    ax[0].set_ylabel('temp. [°C]')
    ax[1].set_ylabel('heat/cool\n power [kW]')
    ax[2].set_ylabel('amb. temp. [°C]')
    ax[-1].set_xlabel('time [h]')


    return fig, ax

# %% 
if __name__ == '__main__':
    
    sid_res = load_sid_results()


    smpc_settings = smpc.base.SMPCSettings(
        prob_chance_cons=.99,
        with_cov=True,
    )
    ms_smpc = setup_controller(sid_res['msm'], smpc_settings)

    test_sys = get_system_for_case_study(sid_res['sigma_x'], sid_res['sigma_y'])

    # %%

    ms_smpc.read_from(test_sys)
    # Simulate the system using the controller for n time steps
    ms_smpc(None, None)

    plot_open_loop_prediction(ms_smpc, sid_res['msm'])

    # %% [markdown]
    """
    Sample the real system with this open-loop prediction
    """
    open_loop_pred_input = sid.InputFromSequence(ms_smpc.res_u_pred)

    data_gen_setup = sid.DataGeneratorSetup(
        N=sid_res['msm'].data_setup.N,
        T_ini=0,
        n_samples = 50,
    )

    data_gen = sid.DataGenerator(
        get_sys = functools.partial(get_system_for_case_study, sid_res['sigma_x'], sid_res['sigma_y']),
        setup = data_gen_setup,
        u_fun = open_loop_pred_input,
    )
    # %%
    fig, ax = plot_open_loop_prediction(ms_smpc, sid_res['msm'])
    for sim_res in data_gen.sim_results:
        ax[0].set_prop_cycle(None)
        ax[0].plot(sim_res.time-3, sim_res.y, alpha=.1)

        ax[-1].set_prop_cycle(None)
        ax[-1].plot(sim_res.time-3, sim_res.x[:,4], alpha=.1)




    # %%
    np.random.seed(int(time.time())*1e6)
# %%
for k in range(10):
    np.random.seed(99)
    print(random.random())
# %%
import random
np.random.seed(99)
# %%
random.random()

# %%
