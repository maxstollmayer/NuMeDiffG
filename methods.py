import numpy as np


# Equation solving methods

def fixed_point_iter(f, y0, *args, tol=0.01, steps=100):
    '''
    Returns approximated fixed point of given function if found using a simple
    fixed point iteration.

    f ....... function with y as first positional argument
    y0 ...... initial value of iteration
    *args ... static arguments of f
    tol ..... tolerance of approximation to stop iterating
    steps ... maximum number of iterations before divergence is declared
    '''

    if isinstance(y0, (int, float)):
        y = np.zeros(steps)
    else:
        y = np.zeros((steps, np.size(y0)))
    y[0] = y0

    for n in range(0, steps):
        y[n+1] = f(y[n], *args)

        if np.allclose(y[n+1], y[n], atol=tol):
            return y[n+1]

    else:
        print(f"Fixed-point iteration did not converge in {steps} steps. Returned initial value.")
        return y0


def newton_raphson(f, df, x0, *args, tol=0.01, steps=100, eps=10**(-10)):
    '''
    Returns approximated root of given function if found using the
    Newton-Raphson method.

    f ....... function with x as first positional argument
    df ...... derivative of f with same arguments as f
    x0 ...... initial value of iteration
    *args ... static arguments of f and df
    tol ..... tolerance of approximation to stop iterating
    steps ... number of values that get calculated
    eps ..... minimum value of the denominator in the calculation
    '''

    if isinstance(x0, (int, float)):
        x = np.zeros(steps)
    else:
        x = np.zeros((steps, np.size(x0)))
    x[0] = x0

    for n in range(0, steps):
        y = f(x[n], *args)
        dy = df(x[n], *args)

        if np.abs(dy) <= eps:
            print(f"Denominator f'(x[{n}]) too small. Returned last value x[{n}].")
            return x[n]

        x[n+1] = x[n] - y / dy

        if np.allclose(x[n+1], x[n], atol=tol):
            return x[n+1]

    else:
        print(f"Newton-Raphson did not converge in {steps} steps. Returned initial value.")
        return x0


def line_method(f, y, *args, tol=0.01, steps=100):
    pass


def dampend_newton(f, y, *args, tol=0.01, steps=100):
    pass


def modified_newton(f, y, *args, tol=0.01, steps=100):
    pass


# ODE solving methods

def forward_euler(f, y0, t0, tN, N):
    '''
    Returns list of input values and list of corresponding function values
    approximated with the forward Euler method.

    f .... function of ODE y' = f(y, t)
    y0 ... initial value y(t0) = y0
    t0 ... starting point of interval
    tN ... end point of interval
    N .... number of steps
    '''

    h = (tN - t0) / N
    t = t0 + h * np.arange(N+1)

    if isinstance(y0, (int, float)):
        y = np.zeros(N+1)
    else:
        y = np.zeros((N+1, np.size(y0)))
    y[0] = y0

    for n in range(0, N):
        y[n+1] = y[n] + h * f(y[n], t[n])

    return t, y


def backward_euler(f, y0, t0, tN, N, tol=0.01):
    '''
    Returns list of input values and list of corresponding function values
    approximated with the backward Euler method.

    f ..... function of ODE y' = f(y, t)
    y0 .... initial value y(t0) = y0
    t0 .... starting point of interval
    tN .... end point of interval
    N ..... number of steps
    tol ... tolerance of approximation to stop iterating
    '''

    h = (tN - t0) / N
    t = t0 + h * np.arange(N+1)

    if isinstance(y0, (int, float)):
        y = np.zeros(N+1)
    else:
        y = np.zeros((N+1, np.size(y0)))
    y[0] = y0

    # function to iterate
    def g(y_iter, t_n, y_n):
        return y_n + h * f(y_iter, t_n)

    for n in range(0, N):
        y[n+1] = fixed_point_iter(g, y[n], t[n], y[n], tol=tol)

    return t, y


def crank_nicolson(f, y0, t0, tN, N, tol=0.01):
    '''
    Returns list of input values and list of corresponding function values
    approximated with the Crank-Nicolson method.

    f ..... function of ODE y' = f(y, t)
    y0 .... initial value y(t0) = y0
    t0 .... starting point of interval
    tN .... end point of interval
    N ..... number of steps
    tol ... tolerance of approximation to stop iterating
    '''

    h = (tN - t0) / N
    t = t0 + h * np.arange(N+1)

    if isinstance(y0, (int, float)):
        y = np.zeros(N+1)
    else:
        y = np.zeros((N+1, np.size(y0)))
    y[0] = y0

    # function to iterate
    def g(y_iter, t_next, y_n, t_n):
        return y_n + h/2 * (f(y_n, t_n) + f(y_iter, t_next))

    for n in range(0, N):
        y[n+1] = fixed_point_iter(g, y[n], t[n+1], y[n], t[n], tol=tol)

    return t, y


def rk4(f, y0, t0, tN, N):
    '''
    Returns list of input values and list of corresponding function values
    approximated with the Runge-Kutta method of 4th order.

    f ..... function of ODE y' = f(y, t)
    y0 .... initial value y(t0) = y0
    t0 .... starting point of interval
    tN .... end point of interval
    N ..... number of steps
    '''

    h = (tN - t0) / N
    t = t0 + h * np.arange(N+1)

    if isinstance(y0, (int, float)):
        y = np.zeros(N+1)
    else:
        y = np.zeros((N+1, np.size(y0)))
    y[0] = y0

    for n in range(0, N):
        k1 = f(y[n], t[n])
        k2 = f(y[n] + h/2 * k1, t[n] + h/2)
        k3 = f(y[n] + h/2 * k1, t[n] + h/2)
        k4 = f(y[n] + h/2 * k1, t[n] + h/2)
        y[n+1] = y[n] + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    return t, y


def adams_bashforth(f, vals, t0, tN, N):
    '''
    Returns list of input values and list of corresponding function values
    approximated with the Adams-Bashforth method.

    f ...... function of ODE y' = f(y, t)
    vals ... initial values (y0, y1, y2, y3)
    t0 ..... starting point of interval
    tN ..... end point of interval
    N ...... number of steps
    '''

    h = (tN - t0) / N
    t = t0 + h * np.arange(N+1)

    if isinstance(vals[0], (int, float)):
        y = np.zeros(N+1)
    else:
        y = np.zeros((N+1, np.size(vals[0])))

    for i in range(0, len(vals)):
        y[i] = vals[i]

    for n in range(0, N-3):
        y[n+4] = (y[n+3] + h/24 * (55*f(y[n+3], t[n+3]) - 59*f(y[n+2], t[n+2])
                                   + 37*f(y[n+1], t[n+1]) - 9*f(y[n], t[n])))

    return t, y


class Runge_Kutta:
    '''
    An instance of Runge_Kutta is an iterative numerical method defined by its
    butcher array.
    '''

    def __init__(self, A, b, c):
        self.order = len(b)
        if A.shape == (self.order, self.order) and len(c) == len(b):
            self.A = A
            self.b = b
            self.c = c
        else:
            raise Exception(f"Input shapes are mismatched. A: {A.shape}, b: {len(b)}, c: {len(c)}")

    def solve(self, f, y0, a, b, h=None, N=None, tol=0.01):
        if not callable(f):
            raise Exception(f"Expected a callable function, not {f}.")
        elif not isinstance(y0, (int, float, list, tuple, np.ndarray)):
            raise Exception(f"Expected a number or array, not {y0}.")
        elif not isinstance(a, (int, float)) or not isinstance(a, (int, float)) or not a < b:
            raise Exception(f"Expected two numbers that satisfy a < b.")
        elif not isinstance(h, (None, int, float)):
            raise Exception("Step size should be a number.")
        elif not isinstance(N, (None, int)):
            raise Exception("Expected a number for the number of steps.")
        elif not isinstance(tol, (float, int)):
            raise Exception("Tolerance should be a number")

        h = (tN - t0) / N
        t = t0 + h * np.arange(N+1)
        y = np.array([y0])

        for n in range(1, N+1):
            self.get_k()

        return t, y

    def get_k(self, f, y_n, t_n):
        '''
        k_j = f(t_n + h * c_j, y_n + h * Sum_l=1^s a_jl * k_l) for j = 1,...,s
        '''

        def g():
            pass

        k = np.zeros(self.s)
        for j in range(1, self.s + 1):
            k[j] = self.iter()

        return k

    def iter(self, f, y0):
        pass
