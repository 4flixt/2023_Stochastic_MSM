
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

# sys.path.append(os.path.join('..'))
# import sysid as sid
# from . import base
# import system
from blrsmpc.sysid import sysid as sid
from . import base
from blrsmpc import system

class StateSpaceSMPC(base.SMPCBase):
    def __init__(self, sid_model: sid.StateSpaceModel, settings: base.SMPCSettings):
        super().__init__(sid_model, settings)
        self._prepare_covariance_propagation()

    def _prepare_covariance_propagation(self):
        sys_A, sys_B, sys_C = system.get_ABC_ARX(
            W = self.sid_model.blr.W.T,
            l = self.sid_model.data_setup.T_ini,
            n_y = self.sid_model.n_y,
            n_u = self.sid_model.n_u,
            )

        self.sys_A = sys_A
        self.sys_B = sys_B
        self.sys_C = sys_C

    def get_covariance(self, x_arx: cas.SX, P0) -> cas.SX:
        Sigma_e = self.sid_model.blr.Sigma_e

        if not self.settings.with_cov:
            Sigma_e = np.diag(np.diag(Sigma_e))

        Sigma_y_new = (Sigma_e*(x_arx.T@self.sid_model.blr.Sigma_p_bar@x_arx)+Sigma_e)

        P_next = self.sys_A@P0@self.sys_A.T + self.sys_C.T@Sigma_y_new@self.sys_C

        Sigma_y_prop = self.sys_C@P_next@self.sys_C.T


        return Sigma_y_prop, P_next

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

        """ Propagate the system dynamics and uncertainty """
        
        y_seq = opt_p['y_past']+opt_x['y_pred']
        u_seq = opt_p['u_past']+opt_x['u_pred']

        T_ini = self.sid_model.data_setup.T_ini
        N = self.sid_model.data_setup.N
        n_y = self.sid_model.n_y

        P0 = np.zeros(self.sys_A.shape)
        Sigma_y_pred = cas.SX.zeros(self.sid_model.n_y*N, self.sid_model.n_y*N)

        for k in range(self.sid_model.data_setup.N):
            xk = cas.vertcat(*y_seq[k:k+T_ini], *u_seq[k:k+T_ini])
            yk_pred = self.sid_model.blr.W.T@xk
            self.cons.add_cons(yk_pred - opt_x['y_pred', k], 0, 0)

            Sigma_y_k, P0 = self.get_covariance(xk, P0)

            Sigma_y_pred[k*n_y:(k+1)*n_y, k*n_y:(k+1)*n_y] = Sigma_y_k
        
         
        """ Constraints """
        y_pred = cas.vertcat(*opt_x['y_pred'])
        for k in range(1, self.sid_model.data_setup.N):
            cons_k = self.stage_cons_fun(opt_x['y_pred', k])
            self.chance_cons.add_cons(cons_k, cons_ub=self._stage_cons.cons_ub, cons_lb=None)

        H = cas.jacobian(self.chance_cons.cons, y_pred)

        cp = np.sqrt(2)*scipy.special.erfinv(2*self.settings.prob_chance_cons -1)
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
            obj += dy_k.T@self.Q@dy_k +  dy_k.T@self.c
            u_k = opt_x['u_pred', k]
            obj += u_k.T@self.R@u_k

            du_k = opt_x['u_pred', k+1] - opt_x['u_pred', k]
            obj += du_k.T@self.delR@du_k


        dy_N = opt_x['y_pred', -1] - opt_p['y_set', -1]
        obj += dy_N.T@self.P@dy_N + dy_N.T@self.c

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
            ct.entry('xk', expr=xk),
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