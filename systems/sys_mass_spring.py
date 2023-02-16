# %%
import sys
import os
from casadi import *
sys.path.append(os.path.join('..', '..', 'do-mpc'))

import do_mpc
# %%

# Define the model
def get_model():

    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.LinearModel(model_type)

    # Same example as shown in the Jupyter Notebooks.

    # Model variables:
    phi_1 = model.set_variable(var_type='_x', var_name='phi_1')
    phi_2 = model.set_variable(var_type='_x', var_name='phi_2')
    phi_3 = model.set_variable(var_type='_x', var_name='phi_3')

    phi = vertcat(phi_1, phi_2, phi_3)

    dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(3,1))

    # Two states for the desired (set) motor position:
    phi_m_set = model.set_variable(var_type='_u', var_name='phi_m_set', shape=(2,1))

    # Two additional states for the true motor position:
    phi_m = model.set_variable(var_type='_x', var_name='phi_m', shape=(2,1))

    # State measurements
    phi_meas = model.set_meas('phi_1_meas', phi)

    Theta_1 = 2.25e-4
    Theta_2 = 2.25e-4
    Theta_3 = 2.25e-4

    c = np.array([2.697,  2.66,  3.05, 2.86])*1e-3
    d = np.array([6.78,  8.01,  8.82])*1e-5


    model.set_rhs('phi_1', dphi[0])
    model.set_rhs('phi_2', dphi[1])
    model.set_rhs('phi_3', dphi[2])

    dphi_next = vertcat(
        -c[0]/Theta_1*(phi[0]-phi_m[0])-c[1]/Theta_1*(phi[0]-phi[1])-d[0]/Theta_1*dphi[0],
        -c[1]/Theta_2*(phi[1]-phi[0])-c[2]/Theta_2*(phi[1]-phi[2])-d[1]/Theta_2*dphi[1],
        -c[2]/Theta_3*(phi[2]-phi[1])-c[3]/Theta_3*(phi[2]-phi_m[1])-d[2]/Theta_3*dphi[2],
    )

    model.set_rhs('dphi', dphi_next)

    tau = 1e-2
    model.set_rhs('phi_m', 1/tau*(phi_m_set - phi_m))

    model.setup()

    discrete_model = model.discretize(t_step = 0.1)

    return discrete_model
    

def get_simulator(model):


    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 0.1)

    simulator.setup()

    return simulator
# %%

model = get_model()
simulator = get_simulator(model)
# %%

simulator.make_step(
    u0=np.array([0, 0]).reshape(-1,1),
    w0=np.ones((8,1)),
    v0=np.ones((3,1))
)
# %%
model.v.shape
# %%
