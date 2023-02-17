
import numpy as np
import scipy
import scipy.io as sio
from scipy.signal import cont2discrete
import sys
import os
import casadi as cas
import casadi.tools as ct
from typing import Union, List, Dict, Tuple, Optional, Callable
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
import pdb

sys.path.append(os.path.join('..'))
import system
import sysid as sid


def chi_sq_pdf(k, x):
    return (x**(k/2-1) * np.exp(-x/2)) / (2**(k/2) * scipy.special.gamma(k/2))

def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + scipy.special.erf((x - mu) / (sigma * np.sqrt(2))))

@dataclass
class SMPCSettings:
    prob_chance_cons: float
    with_cov: bool = True
    nlp_opts: Dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Used for type checking and assertions of the settings
        """
        # Check that prob_chance_cons is a float in (0,1)
        assert isinstance(self.prob_chance_cons, float), "prob_chance_cons must be a float"
        assert self.prob_chance_cons > 0 and self.prob_chance_cons < 1, "prob_chance_cons must be in (0,1)"

        # Check that with_cov is a bool 
        assert isinstance(self.with_cov, bool), "with_cov must be a bool"

    def surpress_ipopt_output(self):
        self.nlp_opts.update({
            'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0
        })

@dataclass
class SMPCFlags:
    TRAINED: bool = False
    SET_OBJECTIVE: bool = False
    SET_CONSTRAINTS: bool = False
    SET_CHANCE_CONSTRAINTS: bool = False
    READ_FROM_SYSTEM: bool = False
    SETUP_NLP: bool = False


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


class SMPCBase:
    def __init__(self, sid_model: Union[sid.MultistepModel, sid.StateSpaceModel], settings: SMPCSettings):
        self.flags = SMPCFlags()
        self.sid_model = sid_model
        self.settings = settings
        self._gen_dummy_variables()

    def _gen_dummy_variables(self):
        self._y_stage = cas.SX.sym('y', self.sid_model.n_y)
        self._u_stage = cas.SX.sym('u', self.sid_model.n_u)
        self._stage_cons = ConstraintHandler()
    
    def set_chance_cons(self, expr: cas.SX, ub: Union[np.ndarray, int, float]):
        expr = expr 

        if not cas.jacobian(expr, self._y_stage).is_constant():
            raise ValueError("Expression must be linear in y and not depend on u")

        self._stage_cons.add_cons(cons=expr, cons_ub=ub)

        self.flags.SET_CHANCE_CONSTRAINTS = True
    
    def set_objective(self, 
                    Q: np.ndarray, 
                    R: Optional[np.ndarray] = None,
                    delR: Optional[np.ndarray] = None, 
                    P: Optional[np.ndarray] = None
                    ):

        if Q.shape != (self.sid_model.n_y, self.sid_model.n_y):
            raise ValueError("Q must be a square matrix with shape (n_y, n_y)")
        if R is None:
            R = np.zeros((self.sid_model.n_u, self.sid_model.n_u))
        elif R.shape != (self.sid_model.n_u, self.sid_model.n_u):
            raise ValueError("R must be a square matrix with shape (n_u, n_u)")
        if delR is None:
            delR = np.zeros((self.sid_model.n_u, self.sid_model.n_u))
        elif delR.shape != (self.sid_model.n_u, self.sid_model.n_u):
            raise ValueError("delR must be a square matrix with shape (n_u, n_u)")
        if P is None:
            P = Q
        elif P.shape != (self.sid_model.n_y, self.sid_model.n_y):
            raise ValueError("P must be a square matrix with shape (n_y, n_y)")

        self.Q = Q
        self.R = R
        self.delR = delR
        self.P = P

        self.flags.SET_OBJECTIVE = True

    
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
        if not self.flags.SETUP_NLP:
            raise ValueError("NLP not setup")

        self.opt_p_num['y_past'] = y_past
        self.opt_p_num['u_past'] = u_past

        self.solve_nlp()

        u_opt = self.opt_x_num['u_pred', 0]

        return u_opt
    
    def read_from(self, system: system.System):
        """ 
        """
        self.system = system
        self.flags.READ_FROM_SYSTEM = True


    @property
    def res_y_pred(self):
        if self.flags.SETUP_NLP:
            res = cas.horzcat(*self.opt_x_num['y_pred']).T.full()
            return res
        else:
            raise ValueError("NLP not setup")
    
    @property
    def res_y_past(self):
        if self.flags.SETUP_NLP:
            res = cas.horzcat(*self.opt_p_num['y_past']).T.full()
            return res
        else:
            raise ValueError("NLP not setup")


    @property
    def res_u_pred(self):
        if self.flags.SETUP_NLP:
            res = cas.horzcat(*self.opt_x_num['u_pred']).T.full()
            return res
        else:
            raise ValueError("NLP not setup")
    
    @property
    def res_u_past(self):
        if self.flags.SETUP_NLP:
            res = cas.horzcat(*self.opt_p_num['u_past']).T.full()
            return res
        else:
            raise ValueError("NLP not setup")

    @property
    def res_y_std(self):
        if self.flags.SETUP_NLP:
            res = np.sqrt(np.diag(self.opt_aux_num['Sigma_y_pred'].full())).reshape(-1, self.sid_model.n_y)
            return res
        else:
            raise ValueError("NLP not setup") 

