# %%
import numpy as np
from typing import Tuple

# %%
def get_ABC() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the matrix triple (A, B, C) of a discrete-time system
    in the form of:

    x[k+1] = A*x[k] + B*u[k]
    y[k]   = C*x[k]

    List of states
        - x1: angle of mass 1
        - x2: angle of mass 2
        - x3: angle of mass 3
        - x4: angular velocity of mass 1
        - x5: angular velocity of mass 2
        - x6: angular velocity of mass 3
        - x7: angle of stepper motor 1 (delayed w.r.t. to inuput u1 with PT1)
        - x8: angle of stepper motor 2 (delayed w.r.t. to inuput u2 with PT1)

    List of inputs
        - u1: angle of stepper motor 1 (setpoint)
        - u2: angle of stepper motor 2 (setpoint)
    """


    A = np.array([[ 8.850e-01,  5.600e-02,  1.000e-03,  9.500e-02,  2.000e-03,
         0.000e+00,  1.000e-02,  0.000e+00],
       [ 5.600e-02,  8.790e-01,  6.400e-02,  2.000e-03,  9.400e-02,
         2.000e-03,  0.000e+00,  0.000e+00],
       [ 1.000e-03,  6.400e-02,  8.740e-01,  0.000e+00,  2.000e-03,
         9.400e-02,  0.000e+00,  1.100e-02],
       [-2.231e+00,  1.071e+00,  2.500e-02,  8.560e-01,  5.500e-02,
         1.000e-03,  1.050e-01,  0.000e+00],
       [ 1.068e+00, -2.338e+00,  1.220e+00,  5.500e-02,  8.450e-01,
         6.300e-02,  5.000e-03,  7.000e-03],
       [ 2.500e-02,  1.217e+00, -2.436e+00,  1.000e-03,  6.300e-02,
         8.370e-01,  0.000e+00,  1.100e-01],
       [ 0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00],
       [ 0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00]])

    B = np.array([[0.048, 0.   ],
       [0.   , 0.   ],
       [0.   , 0.051],
       [1.029, 0.   ],
       [0.017, 0.021],
       [0.   , 1.083],
       [1.   , 0.   ],
       [0.   , 1.   ]])
    
    C = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0]])
    
    return A, B, C
    

# %%
if __name__ == "__main__":
    A, B, C = get_ABC()
    print("A = ", A)
    print("B = ", B)
    print("C = ", C)

    assert A.shape[1] == A.shape[0], "A must be square"
    assert B.shape[0] == A.shape[1], "B must have the same number of rows as A has columns"
    assert C.shape[1] == A.shape[0], "C must have the same number of columns as A has rows"
# %%
