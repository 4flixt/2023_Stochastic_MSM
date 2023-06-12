
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

        W, W0 = self.sid_model._include_scaling_and_bias()

        sys_A, sys_B, sys_C, sys_offset_ARX = system.get_ABC_ARX(
            W = W,
            W0 = W0,
            l = self.sid_model.data_setup.T_ini,
            n_y = self.sid_model.n_y,
            n_u = self.sid_model.n_u,
            )

        self.sys_A = sys_A
        self.sys_B = sys_B
        self.sys_C = sys_C
        self.sys_offset_ARX = sys_offset_ARX
        self.sys_W = W
        self.sys_W0 = W0

        if self.sid_model.blr.state['scale_y']:
            self.S_y = np.diag(self.sid_model.blr.scaler_y.scale_)
        else:
            self.S_y = np.eye(self.sid_model.n_y)

    def _get_covariance(self, x_arx: cas.SX, P0) -> cas.SX:
        Sigma_e = self.sid_model.blr.Sigma_e

        if not self.settings.with_cov:
            Sigma_e = np.diag(np.diag(Sigma_e))

        Sigma_y_new = (Sigma_e*(x_arx.T@self.sid_model.blr.Sigma_p_bar@x_arx)+Sigma_e)
        Sigma_y_new = self.S_y@Sigma_y_new@self.S_y.T


        P_next = self.sys_A@P0@self.sys_A.T + self.sys_C.T@Sigma_y_new@self.sys_C

        Sigma_y_prop = self.sys_C@P_next@self.sys_C.T


        return Sigma_y_prop, P_next

    def _get_y_and_Sigma_y_pred(self, opt_x: ct.struct_symSX, opt_p: ct.struct_symSX) -> Tuple[cas.SX, cas.SX]:
        """ Propagate the system dynamics and uncertainty """
        
        y_seq = opt_p['y_past']+opt_x['y_pred']
        u_seq = opt_p['u_past']+opt_x['u_pred']

        T_ini = self.sid_model.data_setup.T_ini
        N = self.sid_model.data_setup.N
        n_y = self.sid_model.n_y

        P0 = np.zeros(self.sys_A.shape)
        Sigma_y_pred = cas.SX.zeros(self.sid_model.n_y*N, self.sid_model.n_y*N)
        y_pred_calc = cas.SX.zeros(self.sid_model.n_y*N)

        for k in range(self.sid_model.data_setup.N):
            xk = cas.vertcat(*y_seq[k:k+T_ini], *u_seq[k:k+T_ini])
            yk_pred = self.sys_W@xk + self.sys_W0
            Sigma_y_k, P0 = self._get_covariance(xk, P0)

            Sigma_y_pred[k*n_y:(k+1)*n_y, k*n_y:(k+1)*n_y] = Sigma_y_k
            y_pred_calc[k*n_y:(k+1)*n_y] = yk_pred

        return y_pred_calc, Sigma_y_pred

