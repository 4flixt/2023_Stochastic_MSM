import numpy as np
import pandas as pd
import pdb
import casadi as cas
import casadi.tools as ctools
from typing import Callable, Optional, Union, Tuple, List, Dict, Any


_u_fun_type = Callable[[np.ndarray, float], np.ndarray]
 

class System:
    """ A class for simulation discrete-time state-space systems.
    
    """
    def __init__(self, rhs_func, meas_func, x0, u0, dt=1, t_now=0, sig_y=0, sig_x=0):
        try:
            x_next = rhs_func(x0, u0)        
        except:
            raise ValueError("Invalid rhs_func. The rhs_func must take two arguments: x and u.")
        try:
            y = meas_func(x0, u0)        
        except:
            raise ValueError("Invalid meas_func. The meas_func function must take two arguments: x and u.")
        
        if x0.shape != x_next.shape:
            raise ValueError("x0 and x_next must have the same shape.")

        self.rhs_func = rhs_func
        self.meas_func = meas_func 

        self.n_x = x0.shape[0]
        self.n_u = u0.shape[0]
        self.n_y =  y.shape[0]

        if isinstance(sig_y, (float, int)):
            self.sig_y = sig_y*np.ones((self.n_y,1))
        elif isinstance(sig_y, np.ndarray):
            self.sig_y = sig_y.reshape((self.n_y,1)) 
        if isinstance(sig_x, (float, int)):
            self.sig_x = sig_x*np.ones((self.n_x,1))
        elif isinstance(sig_x, np.ndarray):
            self.sig_x = sig_x.reshape((self.n_x,1)) 

        self.dt = dt
        self.reset(x0=x0, t_now=t_now)

    def make_step(self,u):
        """
        Run a simulation step by passing the current input.
        Returns the current measurement y.
        """
        if u.shape != (self.n_u,1):
            raise ValueError("Input must be a column vector of size ({}x1).".format(self.n_u))

        # Store initial x and time
        self._x.append(self.x0)
        self._time.append(self.t_now)

        # Update measurement
        d = self.meas_func(self.x0, u)
        v = self.sig_y*np.random.randn(self.n_y,1)
        y = d + v

        # Update state and time
        self.x0 = self.rhs_func(self.x0, u)
        w = self.sig_x*np.random.randn(self.n_x, 1)
        self.x0 += w

        self.t_now += self.dt

        # Store input and measurement
        self._u.append(u)
        self._y.append(y)
        self._d.append(d)

        return y

    def simulate(self, 
            u_fun: Callable[[np.ndarray, float], np.ndarray], 
            N: int = 1,
            callbacks: Optional[List[Callable]] = None,
            ):
        """
        Simulate the system for N steps using the input function u_fun.
        """
        for k in range(N):
            u = u_fun(self.x0, self.t_now)
            self.make_step(u)
            if callbacks is not None:
                for callback in callbacks:
                    callback(self)

        return self

    def reset(self, x0=None, t_now = 0):
        """Initialize system and clear history.
        """

        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros((self.n_x,1))

        self._x = []
        self._u = []
        self._d = []
        self._y = []
        self.t_now = t_now
        self._time = []


    def narx_io(self, l, delta_y_out = False, return_type = 'numpy'):
        """ Generates NARX input and output data from stored data.
        The NARX input is structured as [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k)].
        The NARX output is structured as [y(k+1)].

        Args:
            l (int): Number of past inputs and outputs to include in the NARX input.

        Returns:
            tuple: (NARX input, NARX output)

        Raises:
            ValueError: If the number of stored inputs and outputs is smaller than l.
            TypeError: If l is not an integer.

        """
        if not isinstance(l, int):
            raise TypeError('l must be an integer.')
        if self.time.size < l:
            raise ValueError("Not enough data to generate NARX input and output. Run make_step at least {} times.".format(l))

        narx_in = []
        for k in range(l):
            narx_in.append(self.y[k:-l+k])
        for k in range(l):
            narx_in.append(self.u[k:-l+k])

        if delta_y_out:
            narx_out = self.y[l:]-self.y[l-1:-1]
        else:
            narx_out = self.y[l:]

        if return_type == 'numpy':
            narx_in = np.concatenate(narx_in, axis=1)

        elif return_type == 'pandas':
            df_y = pd.concat(
                {f'y_{k}' :pd.DataFrame(narx_in[k]) for k in range(l)},
                axis= 1
            )
            df_u = pd.concat(
                {f'u_{i}' :pd.DataFrame(narx_in[l+k]) for i in range(l)},
                axis= 1
            )
            narx_in = pd.concat((df_y, df_u),axis=1)
            narx_out = pd.DataFrame(narx_out)

        elif return_type == 'list':
            pass 
        else:
            raise AttributeError(f'Return type {return_type} is invalid. Must choose from [list, numpy, pandas]')

        return narx_in, narx_out


    @property
    def x(self):
        return np.concatenate(self._x,axis=1).T

    @x.setter
    def x(self, *args):
        raise Exception('Cannot set x directly.')

    @property
    def u(self):
        return np.concatenate(self._u,axis=1).T

    @u.setter
    def u(self, *args):
        raise Exception('Cannot set u directly.')

    @property
    def y(self):
        return np.concatenate(self._y,axis=1).T

    @y.setter
    def y(self, *args):
        raise Exception('Cannot set y directly.')

    @property 
    def d(self):
        return np.concatenate(self._d, axis=1).T

    @d.setter
    def d(self, *args):
        raise Exception('Cannot set d directly.')

    @property
    def time(self):
        return np.array(self._time)

    @time.setter
    def time(self, *args):
        raise Exception('Cannot set time directly.')
    
class LTISystem(System):
    """
    Helper class to simulate linear discrete time dynamical systems in state-space form.
    Initiate with system matrices: A,B,C,D such that:

    x_next = A@x + B@u + w_x
    y      = C@x + D@u + w_y

    Passing D is optional.
    Passing x0 is optional (will results to all zero per default).
    w_x and w_y are zero mean Gaussian noise with standard deviation sig_x and sig_y respectively.

    - Run a simulation step with :py:meth:`make_step` method.
    - Reset (clear history) with :py:meth:`reset` method.
    - Get the current state with :py:attr:`x` attribute (similar for inputs :py:attr:`u` and measurements :py:attr:`y`).

    Args: 
        A (np.ndarray): System matrix of shape (n_x, n_x).
        B (np.ndarray): Input matrix of shape (n_x, n_u).
        C (np.ndarray): Output matrix of shape (n_y, n_x).
        D (np.ndarray): Feedthrough matrix of shape (n_y, n_u). Defaults to None.
        x0 (np.ndarray): Initial state of shape (n_x, 1). Defaults to None.
        dt (float): Time step. Defaults to 1.
        t_now (float): Current time. Defaults to 0.
        sig_y (float): Standard deviation of measurement noise. Defaults to 0.
        sig_x (float): Standard deviation of process noise. Defaults to 0.
    """
    def __init__(self,A,B,C, D=None, x0=None, u0=None, P0=None, dt=1, t_now=0, sig_y = 0, sig_x=0, offset=None):
        self.A = A
        self.B = B
        self.C = C

        if x0 is None: 
            x0 = np.zeros((A.shape[0],1))
        if u0 is None: 
            u0 = np.zeros((B.shape[1],1))


        if D is None:
            self.D = D = np.zeros((C.shape[0], B.shape[1]))
        else:
            self.D = D

        meas_func = get_linear_function(C,D)
        rhs_func = get_linear_function(A,B, offset)

        super().__init__(rhs_func, meas_func, x0=x0, u0=u0, dt=dt, t_now=t_now, sig_y=sig_y, sig_x=sig_x)

        if P0 is None:
            self.P0_x = np.zeros(A.shape)
        else:
            self.P0_x = P0
        self.P0_y = self.C@self.P0_x@self.C.T


        self._P_x = []
        self._P_y = []

    def make_step(self, u, Q=None, E=None, R=None):
        """
        Run a simulation step with input u.

        Args:
            u (np.ndarray): Input of shape (n_u, 1).
        """
        super().make_step(u)


        if self.P0_x is not None:
            self._P_x.append(self.P0_x)
            self._P_y.append(self.P0_y)

            if E is None:
                E = np.eye(self.A.shape[0])
            if Q is None:
                Q = np.eye(self.A.shape[0])
            if R is None:
                R = 0*np.eye(self.C.shape[0])

            P0_x = (self.P0_x + self.P0_x.T)/2 # Make sure P0_x is symmetric

            self.P0_x = self.A@P0_x@self.A.T + E@Q@E.T
            self.P0_y = self.C@P0_x@self.C.T + R

    def simulate(self, 
            u_fun: Callable[[np.ndarray, float], np.ndarray], 
            N: int = 1,
            ):
        """
        Simulate the system for N steps using the input function u_fun.
        """
        Q = self.sig_x**2 * np.eye(self.A.shape[0])
        R = self.sig_y**2 * np.eye(self.C.shape[0])

        for k in range(N):
            u = u_fun(self.x0, self.t_now)
            self.make_step(u, Q=Q, R=R)

        return self

    def reset(self, x0=None, P0=None, t_now=0):
        """
        Reset the system (clear history).
        """
        super().reset(x0=x0, t_now=t_now)
        self.P0_x = P0
        if P0 is not None:
            self.P0_y = self.C@P0@self.C.T
        else:
            self.P0_y = None
        self._P_x = []
        self._P_y = []

    @property
    def P_x(self):
        if self.P0_x is None:
            raise AttributeError('P0 is not set.')
        else:
            return np.stack(self._P_x)

    @property
    def P_y(self):
        if self.P0_y is None:
            raise AttributeError('P0 is not set.')
        else:
            return np.stack(self._P_y)

    def __getstate__(self):
        state = self.__dict__.copy()
        # These cannot be pickled
        state.pop('rhs_func')
        state.pop('meas_func')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rhs_func = get_linear_function(self.A, self.B)
        self.meas_func = get_linear_function(self.C, self.D)


def get_linear_function(A,B, offset=None):
    """
    Returns a function that takes in a state x 
    """
    if offset is None:
        offset = np.zeros((A.shape[0],1))

    def f(x,u):
        return A@x + B@u + offset
    return f

class NARX(System):
    """
    In a NARX system, the state is represented by:
    [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k-1)]
    and for the input we have u(k).

    """

    def __init__(self, narx_func, l, y0, u0, dt=1, t_now=0):

        if not isinstance(l, int):
            raise Exception('l must be an integer.')
        if l<0:
            raise Exception('l must be positive.')

        self.narx_l = l
        self.n_y = y0.shape[0]
        self.n_u = u0.shape[0]

        x0 = np.concatenate(l*[y0] + (l-1)*[u0], axis=0)

        # Splitting index to deconstruct NARX input.
        self.narx_splitter = np.cumsum(l*[self.n_y] + (l-1)*[self.n_u])[:-1]

        def rhs_func(narx_state,u):
            # Split narx state into 
            # [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k-1)]
            narx_state_split = np.split(narx_state, self.narx_splitter, axis=0)
            y_list = narx_state_split[:self.narx_l]
            u_list = narx_state_split[self.narx_l:]
            # Create NARX input by adding u(k) to the end.
            narx_input = np.concatenate((*y_list, *u_list, u), axis=0)
            # Evaluate NARX function
            y_next = narx_func(narx_input)

            # Update NARX state
            y_list.pop(0)
            y_list.append(y_next)

            # Past inputs are only part of the state if l>1
            if l>1:
                u_list.pop(0)
                u_list.append(u)
            else:
                pass

            narx_state_next = np.concatenate((*y_list, *u_list), axis=0)
            # Return updated NARX state
            return narx_state_next

        def meas_func(narx_state, u):
            # Split narx state into 
            # [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k-1)]
            narx_state_split = np.split(narx_state, self.narx_splitter, axis=0)
            # Extract y(k)
            y = narx_state_split[self.narx_l-1]

            return y            
            
        super().__init__(rhs_func, meas_func, x0=x0, u0=u0, dt=dt, t_now=t_now, sig_y=0, sig_x=0) 



class ARX(NARX):
    def __init__(self, 
                W: np.ndarray, 
                l: int, 
                y0: np.ndarray, 
                u0: np.ndarray, 
                dt: float = 1, 
                W0: Optional[np.ndarray] = None, 
                t_now: float = 0.0
            ):
        """
        ARX system of the form:
        ::
            y_k+1 = W @ [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k)]] + W0
            t_k+1 = t_k + dt

        Args:
            W: Weight matrix of shape (n_y, l*(n_y+n_u))
            l: Number of past samples to use in the model.
            y0: Initial output.
            u0: Initial input.
            dt: Time step.
            W0: Constant offset.
            t_now: Initial time.
        """

        if not isinstance(W, np.ndarray):
            raise TypeError(f"W must be a numpy array, but is {type(W)}.")
        if W.ndim != 2:
            raise ValueError(f"W must be a 2D array, but is {W.ndim}D.")
        if W0 is None:
            W0 = np.zeros((W.shape[0],1))
        elif W0.shape != (W.shape[0],1):
            raise ValueError(f"b must have shape ({W.shape[0]}, 1), but has shape {W0.shape}.")

        narx_in_dim = l*(y0.shape[0] + u0.shape[0])
        narx_out_dim = y0.shape[0]

        if not W.shape == (narx_out_dim, narx_in_dim):
            raise ValueError(f"W must have shape ({narx_out_dim}, {narx_in_dim}), but has shape {W.shape}.") 

        self.W = W
        self.W0 = W0

        def narx_func(narx_in):
            return W@narx_in + W0


        super().__init__(narx_func, l=l, y0=y0, u0=u0, dt=dt, t_now=t_now)


    def convert_to_state_space(self, x0=None, u0=None, P0 = None):
        """
        Converts an ARX model to a state space model.
        """
        A_ARX, B_ARX, C_ARX, Offset_ARX = get_ABC_ARX(
            self.W, self.W0, self.narx_l, self.n_y, self.n_u, narx_out_dy = False
        )

        if x0 is None:
            x0 = np.zeros((A_ARX.shape[0],1))
        if u0 is None:
            u0 = np.zeros((B_ARX.shape[1],1))

        return LTISystem(A_ARX, B_ARX, C_ARX, offset = Offset_ARX, x0=x0, u0=u0, P0=P0, dt=self.dt)
        

def get_ABC_ARX(W, W0, l, n_y, n_u, narx_out_dy = False):
    """
    For a (N)ARX model with history length l, state:
    [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k-1)]
    and input u(k), we can write the state equation as:
    
    x(k+1) = A_ARX x(k) + B_ARX u(k)

    assuming that the NARX transition function is linear with:

    y(k+1) = W_x [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k-1)]^T + W_u u(k)

    and W = [W_x, W_u] is the NARX weight matrix, this function returns A_ARX and B_ARX.

    If, the NARX transition function returns the deltay instead of y, we can write the state equation,
    please set narx_out_dy = True.

    """
    x_arx = ctools.struct_symSX([
        ctools.entry("y", shape=(n_y, 1), repeat=l),   # y(k-l), y(k-l+1), ..., y(k)
        ctools.entry("u", shape=(n_u, 1), repeat=l-1), # u(k-l), u(k-l+1), ..., u(k-1)
    ])
    u_arx = cas.SX.sym("u", n_u)                       # u(k)

    arx_in = cas.vertcat(x_arx, u_arx)                 # [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k-1), u(k)]
    arx_out = W@arx_in

    x_arx_next = ctools.struct_SX(x_arx)
    x_arx_next["y", -1] = arx_out

    x_arx_offset = x_arx(0)
    x_arx_offset["y", -1] = W0
    x_arx_offset = x_arx_offset.cat.full()

    if narx_out_dy:
        x_arx_next["y", -1] += x_arx["y", -1]

    for i in range(l-1):
        x_arx_next["y", i] = x_arx["y", i+1]

    # Shift u if l>=2 (otherwise u is not part of the state)
    if l>=2:
        for i in range(l-2):
            x_arx_next["u", i] = x_arx["u", i+1]
        x_arx_next["u", -1] = u_arx

    # Obtain A, B, C matrices through linearization 
    A =cas.Function('a_fun', [x_arx], [cas.jacobian(x_arx_next, x_arx)])(x_arx(0)).full()
    B =cas.Function('b_fun', [x_arx], [cas.jacobian(x_arx_next, u_arx)])(x_arx(0)).full()
    C =cas.Function('c_fun', [x_arx], [cas.jacobian(x_arx, x_arx["y", -1])])(x_arx(0)).full().T


    return A,B,C, x_arx_offset


def test():
    A = np.array([[1,1],[0,1]])
    B = np.array([[0],[1]])
    C = np.array([[1,0]])

    sys = LTISystem(A,B,C, dt=0.1, sig_x=0.1, sig_y=0.1)

def test_arx():
    """
    Test ARX/ NARX system class. 

    1. Create simple LTI system
    2. Generate NARX i/o data from LTI system
    3. Linear regression to estimate ARX system parameter
    4. Create function with linear ARX regression parameters representing the ARX/NARX model
    5. Create NARX system with ARX function
    6. Compare NARX system output to LTI system output
    
    """ 

    # 1. Create simple LTI system
    A = np.array([[ 0.763,  0.460],
              [-0.899,  0.763]])

    B = np.array([[0.014],
                [0.063]])

    C = np.array([[1, 0]])
    D = np.zeros((1,1))

    sig_y = np.array([1e-2])
    sig_x = np.array([1e-2, 1e-2])

    sys = LTISystem(A,B,C,D, sig_y=sig_y, sig_x=sig_x)

    # 2. Generate NARX i/o data from LTI system
    sym_steps = 500
    l = 4
    max_amp = np.pi
    cooldown = 0.2

    for k in range(sym_steps,):
        if k<sym_steps*(1-cooldown):
            u0 = np.random.uniform(-max_amp, max_amp, (1,1))
        else:
            u0 = np.zeros((1,1))
        sys.make_step(u0)

    narx_data = sys.narx_io(l=l, delta_y_out=False, return_type = 'numpy')

    # 3. Linear regression to estimate ARX system parameter
    W = np.linalg.inv(narx_data[0].T@narx_data[0])@narx_data[0].T@narx_data[1]

    # 4. Create function with linear ARX regression parameters representing the ARX/NARX model
    def narx_func(narx_input):
        return W.T@narx_input

    # 5. Create NARX system with ARX function
    narx_sys = NARX(narx_func, l, y0 = np.zeros((1,1)), u0 = np.zeros((1,1)))

    # 6. Compare NARX system output to LTI system output
    sys.reset()
    for k in range(100):
        if k<sym_steps*(1-cooldown):
            u0 = np.random.uniform(-max_amp, max_amp, (1,1))
        else:
            u0 = np.zeros((1,1))
        narx_sys.make_step(u0)
        sys.make_step(u0)
    

    fig, ax = plt.subplots(1,1, sharex=True)

    if narx_sys.y.shape[1] == 1:
        ax = [ax]

    for k in range(narx_sys.y.shape[1]):
        ax[k].plot(narx_sys.y[:,k], label='narx')
        ax[k].plot(sys.y[:,k], label='sys')
    
    ax[0].legend()
    plt.show(block=True)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.io as sio

    test()
    test_arx()