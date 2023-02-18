# %%
import numpy as np
import scipy
import scipy.io as sio
from scipy.signal import cont2discrete
import os
import pickle
import casadi as cas
import casadi.tools as ct
from typing import Union, List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto
import sysid as sid
import system
import copy
import pdb

import matplotlib.pyplot as plt

# %%
def chi_sq_pdf(k, x):
    return (x**(k/2-1) * np.exp(-x/2)) / (2**(k/2) * scipy.special.gamma(k/2))

def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + scipy.special.erf((x - mu) / (sigma * np.sqrt(2))))

@dataclass
class MultiStepMPCSettings:
    prob_chance_cons: float
    with_cov: bool = True


class ConstraintHandler:
    def __init__(self):
        self._cons = []
        self._cons_ub = []
        self._cons_lb = []
        self._cons_ub_cat = np.zeros((0,1))
        self._cons_lb_cat = np.zeros((0,1))

    def add_cons(self, 
                cons: cas.SX, 
                cons_ub: Union[np.ndarray, int ,float] = None, 
                cons_lb: Union[np.ndarray, int, float] = None
                ):

        n_cons = cons.shape[0]

        if cons_ub is None:
            cons_ub = np.inf*np.ones(n_cons)
        elif isinstance(cons_ub, (int, float)):
            cons_ub = np.ones(n_cons)*cons_ub
        elif isinstance(cons_ub, cas.DM):
            cons_ub = cons_ub.full()

        assert cons_ub.shape[0] == n_cons 
        
        if cons_lb is None:
            cons_lb = -np.inf*np.ones(n_cons)
        elif isinstance(cons_lb, (int, float)):
            cons_lb = np.ones(n_cons)*cons_lb
        elif isinstance(cons_lb, cas.DM):
            cons_lb = cons_lb.full()

        assert cons_lb.shape[0] == n_cons

        self._cons.append(cons)
        self._cons_ub.append(cons_ub)
        self._cons_lb.append(cons_lb)

        # Update concatenated constraints 
        self._cons_ub_cat = np.concatenate(self._cons_ub)
        self._cons_lb_cat = np.concatenate(self._cons_lb)

    @property
    def cons(self):
        return cas.vertcat(*self._cons)
    
    @property
    def cons_ub(self):
        return self._cons_ub_cat

    @property
    def cons_lb(self):
        return self._cons_lb_cat


class MultiStepMPC:
    def __init__(self, msm: sid.MultistepModel, settings: MultiStepMPCSettings):
        self.msm = msm
        self.settings = settings
        self.system = None

        self._gen_dummy_variables()
    
    def _gen_dummy_variables(self):
        self._y_stage = cas.SX.sym('y', self.msm.n_y)
        self._u_stage = cas.SX.sym('u', self.msm.n_u)
        self._stage_cons = ConstraintHandler()
    
    def set_chance_cons(self, expr: cas.SX, ub: Union[np.ndarray, int, float]):
        expr = expr 

        if not cas.jacobian(expr, self._y_stage).is_constant():
            raise ValueError("Expression must be linear in y and not depend on u")

        self._stage_cons.add_cons(cons=expr, cons_ub=ub)

    def get_covariance(self, m: cas.SX) -> cas.SX:
        Sigma_e = copy.copy(self.msm.blr.Sigma_e)
        Sigma_e[np.abs(Sigma_e)<1e-9] = 0

        if not self.settings.with_cov:
            Sigma_e = np.diag(np.diag(Sigma_e))

        if self.msm.blr.state['add_bias']:
            m = cas.vertcat(1, m)

        Sigma_y_pred = (Sigma_e*(m.T@self.msm.blr.Sigma_p_bar@m)+ Sigma_e)

        return Sigma_y_pred

    def set_objective(self, 
                    Q: np.ndarray, 
                    R: Optional[np.ndarray] = None,
                    delR: Optional[np.ndarray] = None, 
                    r: Optional[np.ndarray] = None,
                    q: Optional[np.ndarray] = None,
                    P: Optional[np.ndarray] = None
                    ):

        if Q.shape != (self.msm.n_y, self.msm.n_y):
            raise ValueError("Q must be a square matrix with shape (n_y, n_y)")
        if R is None:
            R = np.zeros((self.msm.n_u, self.msm.n_u))
        elif R.shape != (self.msm.n_u, self.msm.n_u):
            raise ValueError("R must be a square matrix with shape (n_u, n_u)")
        if delR is None:
            delR = np.zeros((self.msm.n_u, self.msm.n_u))
        elif delR.shape != (self.msm.n_u, self.msm.n_u):
            raise ValueError("delR must be a square matrix with shape (n_u, n_u)")
        if P is None:
            P = Q
        elif P.shape != (self.msm.n_y, self.msm.n_y):
            raise ValueError("P must be a square matrix with shape (n_y, n_y)")

        self.Q = Q
        self.R = R
        self.delR = delR
        self.P = P


    def setup(self):
        """ 
        """
        stage_cons_fun = cas.Function('stage_cons_fun', [self._y_stage], [self._stage_cons.cons])


        self.cons = ConstraintHandler()
        self.chance_cons = ConstraintHandler()

        n_chance_cons = self._stage_cons.cons.shape[0]

        opt_x = ct.struct_symSX([
            ct.entry("y_pred", shape=self.msm.n_y, repeat=self.msm.data_setup.N),
            ct.entry("u_pred", shape=self.msm.n_u, repeat=self.msm.data_setup.N),
            # ct.entry("eps", shape =n_chance_cons, repeat=self.msm.data_setup.N)
        ])
        opt_p = ct.struct_symSX([
            ct.entry("y_past", shape=self.msm.n_y, repeat=self.msm.data_setup.T_ini),
            ct.entry("u_past", shape=self.msm.n_u, repeat=self.msm.data_setup.T_ini),
            ct.entry("y_set", shape=self.msm.n_y, repeat=self.msm.data_setup.N),
        ])

        m = cas.vertcat(*opt_p['y_past'], *opt_p['u_past'], *opt_x['u_pred'])

        y_pred_calc = self.msm.blr.W.T@m
        y_pred = cas.vertcat(*opt_x['y_pred'])
        Sigma_y_pred = self.get_covariance(m)

        
        """ Constraints """
        sys_cons = cas.vertcat(*opt_x['y_pred']) - y_pred_calc
        self.cons.add_cons(sys_cons, 0, 0)

        for k in range(1, self.msm.data_setup.N):
            cons_k = stage_cons_fun(opt_x['y_pred', k])
            self.chance_cons.add_cons(cons_k, cons_ub=self._stage_cons.cons_ub, cons_lb=None)

        H = cas.jacobian(self.chance_cons.cons, y_pred)

        cp = scipy.special.erfinv(2*self.settings.prob_chance_cons -1)
        self.cp = cp

        for i, H_i in enumerate(cas.vertsplit(H)):
            chance_cons_determ = H_i@y_pred + cp*cas.sqrt(H_i@Sigma_y_pred@H_i.T)
            self.cons.add_cons(chance_cons_determ, cons_ub=self.chance_cons.cons_ub[i], cons_lb=None)

        """ Objective"""
        obj = 0

        du_0 = opt_x['u_pred', 0] - opt_p['u_past', -1]
        obj += du_0.T@self.delR@du_0

        for k in range(self.msm.data_setup.N-1):
            dy_k = opt_x['y_pred', k] - opt_p['y_set', k]
            obj += dy_k.T@self.Q@dy_k
            u_k = opt_x['u_pred', k]
            obj += u_k.T@self.R@u_k

            du_k = opt_x['u_pred', k+1] - opt_x['u_pred', k]
            obj += du_k.T@self.delR@du_k


        dy_N = opt_x['y_pred', -1] - opt_p['y_set', -1]
        obj += dy_N.T@self.P@dy_N

        u_N = opt_x['u_pred', -1]
        obj += u_N.T@self.R@u_N


        """ Bounds """
        self.lb_opt_x = opt_x(-np.inf)
        self.ub_opt_x = opt_x(np.inf)

        nlp = {'x': opt_x, 'p': opt_p, 'f': obj, 'g': self.cons.cons}
        self.solver = cas.nlpsol('solver', 'ipopt', nlp)

        opt_aux_expr = ct.struct_SX([
            ct.entry('Sigma_y_pred', expr=Sigma_y_pred),
            ct.entry('H', expr=H),
            ct.entry('m', expr=m),
        ])

        self.opt_aux_fun = cas.Function('opt_aux_fun', [opt_x, opt_p], [opt_aux_expr])

        self.opt_aux_num = opt_aux_expr(0)
        self.opt_p_num = opt_p(0)
        self.opt_x_num = opt_x(0)

    def solve_nlp(self):
        """ 
        """
        sol = self.solver(
            x0=self.opt_x_num, 
            p=self.opt_p_num, 
            lbx=self.lb_opt_x, 
            ubx=self.ub_opt_x, 
            lbg=self.cons.cons_lb, 
            ubg=self.cons.cons_ub
            )

        self.opt_x_num.master = sol['x']
        self.opt_aux_num.master = self.opt_aux_fun(self.opt_x_num, self.opt_p_num)

    def make_step(self, y_past: List[np.ndarray], u_past: List[np.ndarray]):
        """ 
        """
        self.opt_p_num['y_past'] = y_past
        self.opt_p_num['u_past'] = u_past

        self.solve_nlp()

        u_opt = self.opt_x_num['u_pred', 0]

        return u_opt
    
    def read_from(self, system: system.System):
        """ 
        """
        self.system = system

    def __call__(self, t: float, x: np.ndarray):
        if self.system == None:
            raise ValueError('System not set')
        if self.system.y.shape[0] < self.msm.data_setup.T_ini:
            u_opt = np.zeros((self.msm.n_u, 1))
        else:
            y_list = cas.vertsplit(self.system.y[-self.msm.data_setup.T_ini:])
            u_list = cas.vertsplit(self.system.u[-self.msm.data_setup.T_ini:])

            u_opt = self.make_step(y_list, u_list)
        
        return u_opt

# %%

if __name__ == '__main__':
    load_name = os.path.join('results', 'sid_results', 'mass_spring_prediction_models.pkl')
    with open(load_name, "rb") as f:
        res = pickle.load(f)
        ssm = res['ssm']
        msm = res['msm']


    # %%

    ms_mpc_settings = MultiStepMPCSettings(
        prob_chance_cons=.99,
    )

    ms_mpc = MultiStepMPC(msm, ms_mpc_settings)

    ms_mpc.set_objective(
        Q=np.eye(msm.n_y),
        delR = 1*np.eye(msm.n_u),
        R = 1e-1*np.eye(msm.n_u),
        P=10*np.eye(msm.n_y),
    )

    ms_mpc._y_stage
    ms_mpc.set_chance_cons(expr = ms_mpc._y_stage, ub = 3)

    ms_mpc.setup()


    # %%
    np.random.seed(99)
    sys_generator = sid.SystemGenerator(
            sys_type=sid.SystemType.TRIPLE_MASS_SPRING,
            sig_x=res['sigma_x'],
            sig_y=res['sigma_y'],
            case_kwargs={'state_feedback':False, 'x0': np.zeros((8,1))},
            dt=0.1,
        )

    sys = sys_generator()
    random_input = sid.RandomInput(n_u=2, u_max = 3)

    # Excite for some steps
    sys.simulate(random_input, 100)


    # %%

    y_list = cas.vertsplit(sys.y[-msm.data_setup.T_ini:])
    u_list = cas.vertsplit(sys.u[-msm.data_setup.T_ini:])

    ms_mpc.make_step(y_list, u_list)
    opt_x_num = ms_mpc.opt_x_num
    opt_p_num = ms_mpc.opt_p_num
    opt_aux_num = ms_mpc.opt_aux_num

    # %%
    opt_y_pred = cas.horzcat(*opt_x_num['y_pred']).T.full()
    opt_y_std = np.sqrt(np.diag(opt_aux_num['Sigma_y_pred'].full())).reshape(-1, ssm.n_y)
    opt_y_set = cas.horzcat(*opt_p_num['y_set']).T.full()
    opt_u_pred = cas.horzcat(*opt_x_num['u_pred']).T.full()
    opt_y_past = cas.horzcat(*opt_p_num['y_past']).T.full()
    opt_u_past = cas.horzcat(*opt_p_num['u_past']).T.full()
    fig, ax = plt.subplots(2,1)
    ax[0].plot(np.arange(msm.data_setup.N), opt_y_pred)
    ax[0].set_prop_cycle(None)
    ax[0].plot(np.arange(msm.data_setup.N), opt_y_pred+3*opt_y_std, ':')
    ax[0].set_prop_cycle(None)
    ax[0].plot(np.arange(msm.data_setup.N), opt_y_pred-3*opt_y_std, ':')
    ax[0].set_prop_cycle(None)
    ax[0].plot(np.arange(msm.data_setup.N), opt_y_set, '--')
    ax[0].set_prop_cycle(None)
    ax[0].plot(np.arange(-msm.data_setup.T_ini,0), opt_y_past)

    ax[1].step(np.arange(msm.data_setup.N),opt_u_pred)

    # %% 
    if True:
        ms_mpc.read_from(sys)
        sys.simulate(ms_mpc, 50)
        # %%

        fig, ax = plt.subplots(2,1)
        ax[0].plot(sys.time, sys.y)
        ax[1].step(sys.time, sys.u)
        plt.show(block=True)
    # %%
