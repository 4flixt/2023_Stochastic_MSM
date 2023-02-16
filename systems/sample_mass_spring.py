# %%
import sys
import os
from casadi import *
sys.path.append(os.path.join('..', '..', 'do-mpc'))
import do_mpc
import multiprocessing as mp

import sys_mass_spring


settings = {
    'sample_steps': 500,
    'data_dir_prefix': 'samples_mass_spring'
}

scenarios = [
    {'sigma_x' : 5e-3*np.array([2,1,0.5,0.1,0.1,0.1,0.1,0.1]), 'name': 'small'},
    {'sigma_x' : 1e-2*np.array([2,1,0.5,0.1,0.1,0.1,0.1,0.1]), 'name': 'medium'},
    {'sigma_x' : 5e-2*np.array([2,1,0.5,0.1,0.1,0.1,0.1,0.1]), 'name': 'large'},
]

model = sys_mass_spring.get_model()
simulator = sys_mass_spring.get_simulator(model)

randominput = do_mpc.controller.UniformRandomInput(
    n_u = model.n_u,
    u_lb = -np.pi,
    u_ub = np.pi,
)

def gen_sampling_plan(sigma_x, name):
    np.random.seed(123)

    sp = do_mpc.sampling.SamplingPlanner()
    sp.data_dir = os.path.join(settings['data_dir_prefix'], name, '')

    # Generate sampling function for initial states
    def gen_initial_states():

        x0 = np.random.randn(model.n_x)

        return x0
    
    def gen_e_x():
        e_x = np.random.randn(settings['sample_steps'], model.n_x,1)*sigma_x.reshape(-1,1)
        return e_x 


    # Add variables
    sp.set_sampling_var('x0', gen_initial_states)
    sp.set_sampling_var('e_x', gen_e_x)

    sp.set_param(overwrite=True)

    plan = sp.gen_sampling_plan(n_samples = 20)

    sp.export(f'{name}_noise_mass_spring')

    return plan, sp.data_dir
# %%

def sample_function(x0, e_x):

    simulator.reset_history()

    # set initial values and guess
    x0 = x0
    simulator.x0 = x0


    # run the closed loop for 150 steps
    for k in range(settings['sample_steps']):
        u0 = randominput(x0)
        y0 = simulator.make_step(u0, w0=e_x[k])
    return simulator.data



def generate_samples(plan, data_dir):

    sampler = do_mpc.sampling.Sampler(plan)
    sampler.set_param(overwrite = True)
    sampler.set_param(print_progress = False)
    sampler.data_dir = data_dir


    sampler.set_sample_function(sample_function)


    with mp.Pool(processes=4) as pool:
        p = pool.map(sampler.sample_idx, list(range(sampler.n_samples)))



if __name__ == '__main__':
    for scenario in scenarios:
        plan, data_dir = gen_sampling_plan(**scenario)
        generate_samples(plan, data_dir)

# %%
randominput()
# %%
randominput.u_ub
# %%
