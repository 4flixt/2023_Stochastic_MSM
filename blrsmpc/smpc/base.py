
import numpy as np
import scipy
import casadi as cas
import casadi.tools as ct
from typing import Union, List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pdb

from blrsmpc import system
from blrsmpc.sysid import sysid as sid


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


class SMPCBase(ABC):
    """
    Base class for SMPC with identified system model.
    Identification is done with either a state-space model or a multi-step model.
    
    """
    def __init__(self, sid_model: Union[sid.MultistepModel, sid.StateSpaceModel], settings: SMPCSettings):
        self.flags = SMPCFlags()
        self.sid_model = sid_model
        self.settings = settings
        self._gen_dummy_variables()

    def _gen_dummy_variables(self):
        self._y_stage    = cas.SX.sym('y', self.sid_model.n_y)
        self._y_setpoint = cas.SX.sym('y_sp', self.sid_model.n_y)
        self._u_stage    = cas.SX.sym('u', self.sid_model.n_u)
        self._u_previous = cas.SX.sym('u_prev', self.sid_model.n_u)
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
                    P: Optional[np.ndarray] = None,
                    c: Optional[np.ndarray] = None,
                    ):
    
        """
        Define quadratic objective function for the MPC problem. The objective function is defined as:

        ::

            dx = (x - x_s)
            du = (u - u_prev)

            J(x,u) = dx.T@Q@dx + c.T@x + u.T@R@u + du.T@delR@du 

            m(x) = dx.T@P@dx + c.T@x
        
        """

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
        if c is None:
            c = np.zeros((self.sid_model.n_y, 1))
        elif c.shape != (self.sid_model.n_y, 1):
            raise ValueError("c must be a column vector with shape (n_y, 1)")

        self.Q = Q
        self.R = R
        self.delR = delR
        self.P = P
        self.c = c

        dy = self._y_stage - self._y_setpoint
        du = self._u_stage - self._u_previous

        stage_term = dy.T@Q@dy + c.T@self._y_stage + du.T@R@du + du.T@delR@du
        terminal_term = dy.T@P@dy + c.T@self._y_stage

        stage_cost = cas.Function('stage_cost', 
                                    [self._y_stage, self._u_stage, self._u_previous, self._y_setpoint], 
                                    [stage_term]
                                  )
        terminal_cost = cas.Function('terminal_cost',
                                    [self._y_stage, self._y_setpoint],
                                    [terminal_term]
                                    )
        
        self.set_objective_fun(stage_cost, terminal_cost)


    def set_objective_fun(self, stage_cost: cas.Function, terminal_cost: cas.Function):
        """
        Define custom objective function. The CasADi functions must have the following signature:

        - stage_cost: ``cas.Function([y_stage, u_stage, u_previous, y_setpoint], [stage_term])``
        - terminal_cost: ``cas.Function([y_stage, y_setpoint], [terminal_term])`` 

        """
        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost
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

    def __call__(self, t: float, x: np.ndarray):
        if not self.flags.READ_FROM_SYSTEM:
            raise ValueError('System not set')

        if self.system.y.shape[0] < self.sid_model.data_setup.T_ini:
            u_opt = np.zeros((self.sid_model.n_u, 1))
        else:
            y_list = cas.vertsplit(self.system.y[-self.sid_model.data_setup.T_ini:])
            u_list = cas.vertsplit(self.system.u[-self.sid_model.data_setup.T_ini:])

            u_opt = self.make_step(y_list, u_list)
        
        return u_opt
    
    def read_from(self, system: system.System):
        """ 
        """
        self.system = system
        self.flags.READ_FROM_SYSTEM = True

    @abstractmethod
    def _get_y_and_Sigma_y_pred(self, opt_x: ct.struct_symSX, opt_p: ct.struct_symSX) -> Tuple[cas.SX, cas.SX]:
        """
        Implement this method to return the predicted output and covariance matrix.
        The implementation differs for the state-space and the multi-step model.
        """
        pass

    def setup(self):
        """ 
        """
        self.stage_cons_fun = cas.Function('stage_cons_fun', [self._y_stage], [self._stage_cons.cons])


        self.cons = ConstraintHandler()
        self.chance_cons = ConstraintHandler()

        opt_x = ct.struct_symSX([
            ct.entry("y_pred", shape=self.sid_model.n_y, repeat=self.sid_model.data_setup.N),
            ct.entry("u_pred", shape=self.sid_model.n_u, repeat=self.sid_model.data_setup.N),
        ])
        opt_p = ct.struct_symSX([
            ct.entry("y_past", shape=self.sid_model.n_y, repeat=self.sid_model.data_setup.T_ini),
            ct.entry("u_past", shape=self.sid_model.n_u, repeat=self.sid_model.data_setup.T_ini),
            ct.entry("y_set", shape=self.sid_model.n_y, repeat=self.sid_model.data_setup.N),
        ])

        y_pred = cas.vertcat(*opt_x['y_pred'])
        y_pred_calc, Sigma_y_pred = self._get_y_and_Sigma_y_pred(opt_x, opt_p)

        
        """ Constraints """
        sys_cons = cas.vertcat(*opt_x['y_pred']) - y_pred_calc
        self.cons.add_cons(sys_cons, 0, 0)

        for k in range(1, self.sid_model.data_setup.N):
            cons_k = self.stage_cons_fun(opt_x['y_pred', k])
            self.chance_cons.add_cons(cons_k, cons_ub=self._stage_cons.cons_ub, cons_lb=None)

        H = cas.jacobian(self.chance_cons.cons, y_pred)

        cp = np.sqrt(2)*scipy.special.erfinv(2*self.settings.prob_chance_cons -1)
        self.cp = cp

        for i, H_i in enumerate(cas.vertsplit(H)):
            chance_cons_determ = H_i@y_pred + cp*cas.sqrt(H_i@Sigma_y_pred@H_i.T)
            self.cons.add_cons(chance_cons_determ, cons_ub=self.chance_cons.cons_ub[i], cons_lb=None)

        """ Objective """
        obj = 0
        for k in range(self.sid_model.data_setup.N):
            if k == 0:
                u_prev = opt_p['u_past', -1]
            else:
                u_prev = opt_x['u_pred', k-1]

            y_k = opt_x['y_pred', k]
            u_k = opt_x['u_pred', k]
            y_set_k = opt_p['y_set', k]

            obj += self.stage_cost(y_k, u_k, u_prev, y_set_k)

        """ Bounds """
        self.lb_opt_x = opt_x(-np.inf)
        self.ub_opt_x = opt_x(np.inf)

        nlp = {'x': opt_x, 'p': opt_p, 'f': obj, 'g': self.cons.cons}
        self.solver = cas.nlpsol('solver', 'ipopt', nlp, self.settings.nlp_opts)

        opt_aux_expr = ct.struct_SX([
            ct.entry('Sigma_y_pred', expr=Sigma_y_pred),
            ct.entry('H', expr=H),
        ])

        self.opt_aux_fun = cas.Function('opt_aux_fun', [opt_x, opt_p], [opt_aux_expr])

        self.opt_aux_num = opt_aux_expr(0)
        self.opt_p_num = opt_p(0)
        self.opt_x_num = opt_x(0)

        self.opt_x = opt_x

        self.flags.SETUP_NLP = True


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

