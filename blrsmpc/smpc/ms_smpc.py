
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
from blrsmpc.sysid import sysid as sid
from . import base

class MultiStepSMPC(base.SMPCBase):
    def __init__(self, sid_model: sid.MultistepModel, settings: base.SMPCSettings):
        super().__init__(sid_model, settings)
    
    
    def _get_y_and_Sigma_y_pred(self, opt_x: ct.struct_symSX, opt_p: ct.struct_symSX) -> Tuple[cas.SX, cas.SX]:

        # Prepare m
        m = cas.vertcat(*opt_p['y_past'], *opt_p['u_past'], *opt_x['u_pred'])

        if self.sid_model.blr.state['scale_x']:
            mu_x = self.sid_model.blr.scaler_x.mean_
            s_x = self.sid_model.blr.scaler_x.scale_

            m = (m - mu_x)/s_x

        if self.sid_model.blr.state['add_bias']:
            m = cas.vertcat(m, 1)

       # Prepare Sigma_e
        Sigma_e = copy.copy(self.sid_model.blr.Sigma_e)
        Sigma_e[np.abs(Sigma_e)<1e-9] = 0

        if not self.settings.with_cov:
            Sigma_e = np.diag(np.diag(Sigma_e)) 

        # Prediction
        y_pred_scaled = self.sid_model.blr.W.T@m
        Sigma_y_pred_scaled = (Sigma_e*(m.T@self.sid_model.blr.Sigma_p_bar@m)+ Sigma_e)


        # Unscale if needed
        if self.sid_model.blr.state['scale_y']:
            S_y = np.diag(self.sid_model.blr.scaler_y.scale_)
            y_pred = S_y@y_pred_scaled + self.sid_model.blr.scaler_y.mean_
            Sigma_y_pred = S_y@Sigma_y_pred_scaled@S_y.T
        else:
            Sigma_y_pred = Sigma_y_pred_scaled
            y_pred = y_pred_scaled

        return y_pred, Sigma_y_pred

