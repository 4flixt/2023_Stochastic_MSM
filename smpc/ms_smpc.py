
import numpy as np
import scipy
import scipy.io as sio
import sys
import os
import casadi as cas
import casadi.tools as ct
from typing import Union, List, Dict, Tuple, Optional, Callable
from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
import copy
import pdb

sys.path.append(os.path.join('..'))
import sysid as sid
from . import base

class MultiStepSMPC(base.SMPCBase):
    def __init__(self, sid_model: sid.MultistepModel, settings: base.SMPCSettings):
        super().__init__(sid_model, settings)
    
    def get_covariance(self, m: cas.SX) -> cas.SX:
        Sigma_e = copy.copy(self.sid_model.blr.Sigma_e)
        Sigma_e[np.abs(Sigma_e)<1e-9] = 0

        if not self.settings.with_cov:
            Sigma_e = np.diag(np.diag(Sigma_e))

        if self.sid_model.blr.state['add_bias']:
            m = cas.vertcat(1, m)

        Sigma_y_pred = (Sigma_e*(m.T@self.sid_model.blr.Sigma_p_bar@m)+ Sigma_e)

        return Sigma_y_pred


    def setup(self):
        """ 
        """
        self.stage_cons_fun = cas.Function('stage_cons_fun', [self._y_stage], [self._stage_cons.cons])


        self.cons = base.ConstraintHandler()
        self.chance_cons = base.ConstraintHandler()

        opt_x = ct.struct_symSX([
            ct.entry("y_pred", shape=self.sid_model.n_y, repeat=self.sid_model.data_setup.N),
            ct.entry("u_pred", shape=self.sid_model.n_u, repeat=self.sid_model.data_setup.N),
        ])
        opt_p = ct.struct_symSX([
            ct.entry("y_past", shape=self.sid_model.n_y, repeat=self.sid_model.data_setup.T_ini),
            ct.entry("u_past", shape=self.sid_model.n_u, repeat=self.sid_model.data_setup.T_ini),
            ct.entry("y_set", shape=self.sid_model.n_y, repeat=self.sid_model.data_setup.N),
        ])

        m = cas.vertcat(*opt_p['y_past'], *opt_p['u_past'], *opt_x['u_pred'])

        y_pred_calc = self.sid_model.blr.W.T@m
        y_pred = cas.vertcat(*opt_x['y_pred'])
        Sigma_y_pred = self.get_covariance(m)

        
        """ Constraints """
        sys_cons = cas.vertcat(*opt_x['y_pred']) - y_pred_calc
        self.cons.add_cons(sys_cons, 0, 0)

        for k in range(1, self.sid_model.data_setup.N):
            cons_k = self.stage_cons_fun(opt_x['y_pred', k])
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

        for k in range(self.sid_model.data_setup.N-1):
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
        self.solver = cas.nlpsol('solver', 'ipopt', nlp, self.settings.nlp_opts)

        opt_aux_expr = ct.struct_SX([
            ct.entry('Sigma_y_pred', expr=Sigma_y_pred),
            ct.entry('H', expr=H),
            ct.entry('m', expr=m),
        ])

        self.opt_aux_fun = cas.Function('opt_aux_fun', [opt_x, opt_p], [opt_aux_expr])

        self.opt_aux_num = opt_aux_expr(0)
        self.opt_p_num = opt_p(0)
        self.opt_x_num = opt_x(0)

        self.flags.SETUP_NLP = True

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