
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as sio
from scipy.signal import cont2discrete
from casadi import *
from casadi.tools import *
from typing import Union, List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto

import system
import bayli
import helper
# %%
class SystemType(Enum):
    TRIPLE_MASS_SPRING = auto()
    BUILDING = auto()

class SystemGenerator:
    """ Used to configure a system and generate new instances of it.
    """
    def __init__(self,
            sys_type: Optional[SystemType] = SystemType.TRIPLE_MASS_SPRING,
            sig_x: Union[float, np.ndarray] = 0.0,
            sig_y: Union[float, np.ndarray] = 0.0,
            dt: float = 0.1,
            case_kwargs: Optional[Dict] = dict(),
        ):
        self.sys_type = sys_type
        self.sig_x = sig_x
        self.sig_y = sig_y
        self.dt = dt
        self.case_kwargs = case_kwargs


    def triple_mass_spring(self, state_feedback=False, x0=None):
        A, B, C = system.triple_mass_spring.get_ABC()
        D = np.zeros((C.shape[0], B.shape[1]))

        if state_feedback:
            C = np.eye(8)
            D = np.zeros((8,2))

        if x0 is None:
            x0 = np.pi*np.random.randn(8,1)

        sys = system.LTISystem(A,B,C,D, x0=x0, sig_x=self.sig_x, sig_y=self.sig_y, dt=self.dt)

        return sys

    def building_system(self, x0=None, state_feedback=True):
        A, B, C = system.building_system.get_ABC()

        if x0 is None:
            t_build = np.random.uniform(14,27, size=(4,1))
            to = np.random.uniform(0,30, size=(1,1))
            x0 = np.concatenate([t_build, to], axis=0)

        if not state_feedback:
            C = np.array([
                [1,0,0,0,0],
                [0,1,0,0,0],
                [0,0,1,0,0],
                [0,0,0,1,0],
            ])

        sys = system.LTISystem(A, B, C, x0=x0, sig_x=self.sig_x, sig_y=self.sig_y, dt=self.dt)

        return sys


    def __call__(self):
        if self.sys_type == SystemType.TRIPLE_MASS_SPRING:
            return self.triple_mass_spring(**self.case_kwargs)
        elif self.sys_type == SystemType.BUILDING:
            return self.building_system(**self.case_kwargs)
        else:
            raise NotImplementedError('Unknown system type.')


class RandomInput:
    def __init__(self, 
        n_u: int, 
        switch_prob: Optional[float] =0.5, 
        u_max: Optional[float] = None,
        u_lb: Optional[float] = None,
        u_ub: Optional[float] = None,
        u0: Optional[float] = None,
    ):
        self.n_u = n_u
        if u0 is None:
            self.u = np.zeros((self.n_u, 1))
        else:
            self.u = u0
        
        self.switch_prob = switch_prob
        
        if u_max is not None:
            self.u_lb = -u_max
            self.u_ub = u_max
        elif u_lb is not None and u_ub is not None:
            self.u_lb = u_lb
            self.u_ub = u_ub
        else:
            raise ValueError('Must specify either u_max or u_lb and u_ub.')


    def __call__(self, x, t):
        u_candidate = np.random.uniform(self.u_lb, self.u_ub)
        switch = np.random.rand(self.n_u, 1) >= (1-self.switch_prob) # switching? 0 or 1.
        self.u = (1-switch)*self.u + switch*u_candidate # Old or new value.
        return self.u

    def gen_sequence(self, T):
        input_sequence = np.zeros((T, self.n_u))

        for t in range(T):
            input_sequence[t,:] = self(None, None).reshape(-1)

        return input_sequence

class InputFromSequence:
    def __init__(self, 
            sequence: np.ndarray,
            infinite_loop: Optional[bool] = True
        ):
        self.sequence = sequence
        self.m, self.n_u = sequence.shape
        self.infinite_loop = infinite_loop
        self.running_index = 0

    def __call__(self, x: Optional[np.ndarray] = None, t: Optional[float] = None):
        if self.infinite_loop: 
            k = int(mod(self.running_index, self.m))
        elif self.running_index < self.m:
            k = self.running_index
        else:
            raise ValueError('Input sequence is not infinite and has been exhausted.')

        u = self.sequence[k,:].reshape(-1,1)
        self.running_index += 1
        return u

# %%
@dataclass
class DataGeneratorSetup:
    T_ini: int
    N: int
    n_samples: int
    sig_x: Union[float, np.ndarray] = 0.0
    sig_y: Union[float, np.ndarray] = 0.0
    dt: float = 0.1

    @property
    def L(self):
        return self.T_ini + self.N

class DataGenerator:
    def __init__(self,
            get_sys: Callable[[], system.LTISystem],
            setup: DataGeneratorSetup,
            u_fun: Callable[[np.ndarray, float], np.ndarray],  
        ):

        self.setup = setup
        self.sim_results = []

        for k in range(setup.n_samples):
            sys = get_sys()
            self.sim_results.append(sys.simulate(u_fun, setup.L))
            helper.print_percent_done(k, setup.n_samples, title='Sampling data...')

    def get_narx_io(self, **kwargs):
        narx_in = []
        narx_out = []

        for sim_result in self.sim_results:
            in_k, out_k = sim_result.narx_io(self.setup.T_ini, **kwargs)
            narx_in.append(in_k)
            narx_out.append(out_k)

        narx_in = np.concatenate(narx_in, axis=0)
        narx_out = np.concatenate(narx_out, axis=0)

        return narx_in, narx_out

    @property
    def U_T_ini(self):
        return np.concatenate([r.u[:self.setup.T_ini].reshape(-1,1) for r in self.sim_results], axis=1)
    @property
    def Y_T_ini(self):
        return np.concatenate([r.y[:self.setup.T_ini].reshape(-1,1) for r in self.sim_results], axis=1)
    @property
    def X_T_ini(self):
        return np.concatenate([r.x[:self.setup.T_ini].reshape(-1,1) for r in self.sim_results], axis=1)
    @property
    def U_N(self):
        return np.concatenate([r.u[self.setup.T_ini:].reshape(-1,1) for r in self.sim_results], axis=1)
    @property
    def Y_N(self):
        return np.concatenate([r.y[self.setup.T_ini:].reshape(-1,1) for r in self.sim_results], axis=1)
    @property
    def X_N(self):
        return np.concatenate([r.x[self.setup.T_ini:].reshape(-1,1) for r in self.sim_results], axis=1)
    @property
    def U_L(self):
        return np.concatenate((self.U_T_ini, self.U_N), axis=0)
    @property
    def Y_L(self):
        return np.concatenate((self.Y_T_ini, self.Y_N), axis=0)
    @property
    def X_L(self):
        return np.concatenate((self.X_T_ini, self.X_N), axis=0)
    @property
    def M(self):
        return np.concatenate((self.Y_T_ini, self.U_L), axis=0)
    @property
    def n_y(self):
        return self.sim_results[0].n_y
    @property
    def n_u(self):
        return self.sim_results[0].n_u
    @property
    def n_x(self):
        return self.sim_results[0].n_x

    
class MultistepModel:
    def __init__(
            self,
            **kwargs
        ):
        # Bias cannot be estimated.
        kwargs.update(add_bias=False)
        self.blr = bayli.BayesianLinearRegression(
            **kwargs
        )

        # if self.blr.scale_x or self.blr.scale_y or self.blr.add_bias:
        #     raise ValueError('Multi-step model does not support BLR with scaling or bias.')

    def get_sparsity(self, data: DataGenerator):
        mat_sparsity_sigma_e = np.kron(np.eye(data.setup.N), np.tril(np.ones((data.n_y, data.n_y)))) == 1
        srow, scol = np.where(mat_sparsity_sigma_e)
        nrow, ncol = mat_sparsity_sigma_e.shape

        sigma_e_sparsity = Sparsity.triplet(nrow, ncol, list(srow), list(scol))

        sparsity_matrix_T_ini = np.kron(np.ones((data.setup.N, data.setup.T_ini)), np.ones((data.n_y, data.n_u+data.n_y))).astype(bool)
        sparsity_matrix_N = np.kron(np.tril(np.ones((data.setup.N, data.setup.N))), np.ones((data.n_y, data.n_u))).astype(bool)
        sparsity_matrix = np.concatenate((sparsity_matrix_T_ini, sparsity_matrix_N), axis=1).astype(bool)
        sparsity_matrix = sparsity_matrix.T

        srow, scol = np.where(sparsity_matrix)
        nrow, ncol = sparsity_matrix.shape
        w_sparsity = Sparsity.triplet(nrow, ncol, list(srow), list(scol))

        return sigma_e_sparsity, w_sparsity

    def fit(self, data: DataGenerator, **kwargs):
        self.data_setup = data.setup

        self.n_u = data.n_u
        self.n_y = data.n_y

        sigma_e_sparsity, w_sparsity = self.get_sparsity(data)
        self.blr.fit(data.M.T, data.Y_N.T, w_sparsity=w_sparsity, sigma_e_sparsity=sigma_e_sparsity, **kwargs)

    def predict(self, *args, **kwargs):
        return self.blr.predict(*args, **kwargs)


class StateSpaceModel:
    def __init__(
            self,
            **kwargs
            ):
        # Bias cannot be estimated.
        kwargs.update(add_bias=False)
        self.blr = bayli.BayesianLinearRegression(
            **kwargs
            )

        # if self.blr.scale_x or self.blr.scale_y or self.blr.add_bias:
        #     raise ValueError('State space model does not support BLR with scaling or bias.')

    def fit(self, data: DataGenerator):

        self.data_setup = data.setup
        self.n_u = data.n_u
        self.n_y = data.n_y

        narx_in, narx_out = data.get_narx_io()
        self.blr.fit(narx_in, narx_out)

        y0 = np.zeros((data.n_y, 1))
        u0 = np.zeros((data.n_u, 1))

        arx = system.ARX(self.blr.W.T, l=self.data_setup.T_ini, y0=y0, u0=u0, dt=self.data_setup.dt)
        self.LTI = arx.convert_to_state_space()

    def predict(self, *args, **kwargs):
        return self.blr.predict(*args, **kwargs)


    def predict_sequence(self, m, **kwargs):
        """ Harmonized interface with MSM."""

        number_input_predictions = (self.data_setup.N+1 )* self.n_u

        x0= m[:-number_input_predictions].reshape(-1,1)
        input_sequence = m[-number_input_predictions:].reshape(-1,self.n_u)


        self.LTI.reset(x0=x0, P0=np.zeros(self.LTI.A.shape))

        for u in input_sequence:
            u = u.reshape(-1,1)
            arx_in = np.concatenate((self.LTI.x0, u), axis=0)
            _, Q = self.predict(arx_in.T, uncert_type='cov', **kwargs)

            self.LTI.make_step(u, Q, self.LTI.C.T)

        x_seq = self.LTI.x[1:] # Remove initial condition
        P_seq = self.LTI.P[1:] # Remove initial condition
        C = self.LTI.C

        y = x_seq@C.T
        std = np.sqrt((np.diagonal(P_seq, axis1=1, axis2=2)@C.T))

        return y, std
