
# %%
import numpy as np
from scipy.signal import cont2discrete
from typing import Tuple, Optional

# %%
def get_ABC(t_samp: Optional[float]= 3600.00) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the matrix triple (A, B, C) of a discrete-time system
    in the form of:

    x[k+1] = A*x[k] + B*u[k]
    y[k]   = C*x[k]

    Args:
        t_samp: sampling time in seconds

    List of states:
        - x1: temp [°C] of room 1
        - x2: temp [°C] of room 2
        - x3: temp [°C] of room 3
        - x4: temp [°C] of room 4
        - x5: ambient temp [°C]

    List of inputs:
        - heat/cool [kW] of room 1
        - heat/cool [kW] of room 2
        - heat/cool [kW] of room 3
        - heat/cool [kW] of room 4
    """

    C = np.diag([50, 110, 80, 90])*1e3 # [kJ/K]

    H = np.array([
    [0.0, 2.1, 2.0, 0.0, 0.3],
    [2.1, 0.0, 0.0, 1.9, 0.5],
    [2.0, 0.0, 0.0, 1.0, 0.4],
    [0.0, 1.9, 1.0, 0.0, 0.6]
    ])               # [kW/K]

    D = np.concatenate([
        np.diag(np.sum(H, axis=1)),
        np.zeros((4,1))
    ], axis=1)

    ts = 20*3600 #s
    a0 = np.array([0,0,0,0, 1/ts]).reshape(1,-1)

    A = np.concatenate([np.linalg.inv(C)@(H-D), -a0], axis=0)
    B = np.concatenate([np.linalg.inv(C), np.zeros((1,4))], axis=0)
    B = np.concatenate([B, a0.T], axis=1)
    C = np.eye(5)
    D = np.zeros((5,5))

    A_dc, B_dc, C_dc, D_dc, _ = cont2discrete((A, B, C, D), t_samp, method='zoh')

    return A_dc, B_dc, C_dc  

# %%

if __name__ == "__main__":
    A, B, C = get_ABC()
    print(A)
    print(B)
    print(C)

    assert A.shape[1] == A.shape[0], "A must be square"
    assert B.shape[0] == A.shape[1], "B must have the same number of rows as A has columns"
    assert C.shape[1] == A.shape[0], "C must have the same number of columns as A has rows"

    C = np.array([
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,0,0,1],
    ])

    # observability matrix
    O = np.concatenate([C@np.linalg.matrix_power(A, i) for i in range(2)], axis=0)


    print(np.linalg.eig(A)[0])
# %%
