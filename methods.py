import numpy as np


def forward_euler(f, y0, t0, tN, N):
    '''
    Returns list of input values and list of corresponding function values
    approximated with the forward Euler method

    f is the function of the ODE y' = f(y, t)
    t0 and tN are the boundaries of the interval [t0, tN]
    this interval is divided into N+1 steps
    y0 is the initial condition y(t0) = y0
    '''

    # step size
    h = (tN - t0) / N

    # list of input values with step size h
    t = t0 + h * np.arange(N+1)

    # array of function vectors starting with the initial value
    y = np.array([y0])

    # loop to generate function values with forward euler method
    for n in range(0, N):
        new = y[n] + h * f(y[n], t[n])
        y = np.append(y, [new], axis=0)

    return t, y


def fixed_point_iter(f, y, *args, tol=0.01, steps=100):
    '''
    Nests given function f on first positional argument y from starting value
    until desired accuracy is reached
    otherwise returns last y value after N steps
    '''

    # compute new value
    new = f(y, *args)

    # if desired accuracy is reached, returns new value
    if np.allclose(y, new, atol=tol):
        return new

    # returns last y value after n steps if accuracy is not reached
    elif steps == 0:
        print(f"Fixed-point iteration did not converge for {f}.")
        return y

    # if neither accuracy nor step limit is reached,
    # then the iteration continues recursively
    else:
        return fixed_point_iter(f, new, *args, tol=tol, steps=steps-1)


def backward_euler(f, y0, t0, tN, N, tol=0.001):
    '''
    Returns list of input values and list of corresponding function values
    approximated with the backward Euler method

    f is the function of the ODE y' = f(y, t)
    t0 and tN are the boundaries of the interval [t0, tN]
    this interval is divided into N+1 steps
    y0 is the initial condition y(t0) = y0
    tol is the tolerance for convergence of the fixed point iteration
    '''
    # step size
    h = (tN - t0) / N

    # list of input values with step size h
    t = t0 + h * np.arange(N+1)

    # array of function vectors starting with the initial value
    y = np.array([y0])

    # function of the backward Euler method for the iteration
    # y_iter is the value that gets iterated, t_n and y_n are static
    def g(y_iter, t_n, y_n):
        return y_n + h * f(y_iter, t_n)

    # loop to calculate values for all inputs t
    # fixed point iteration is applied to find approximation of next y value
    for n in range(0, N):
        new = fixed_point_iter(g, y[n], t[n], y[n], tol=tol)
        y = np.append(y, [new], axis=0)

    return t, y


def crank_nicolson(f, y0, t0, tN, N, tol=0.001):
    '''
    Returns list of input values and list of corresponding function values
    approximated with the Crank-Nicolson method

    f is the function of the ODE y' = f(y, t)
    t0 and tN are the boundaries of the interval [t0, tN]
    this interval is divided into N+1 steps
    y0 is the initial condition y(t0) = y0
    tol is the tolerance for convergence of the fixed point iteration
    '''

    # step size
    h = (tN - t0) / N

    # list of input values with step size h
    t = t0 + h * np.arange(N+1)

    # array of function vectors starting with the initial value
    y = np.array([y0])

    # function of the Crank-Nicolson method for the iteration
    # y_iter gets iterated, t_next, y_n, t_n are static
    def g(y_iter, t_next, y_n, t_n):
        return y_n + h/2 * (f(y_n, t_n) + f(y_iter, t_next))

    # loop to calculate values for all inputs t
    # fixed point iteration is applied to find approximation for next y value
    for n in range(0, N):
        new = fixed_point_iter(g, y[n], t[n+1], y[n], t[n], tol=tol)
        y = np.append(y, [new], axis=0)

    return t, y


class Runge_Kutta:
    '''
    An instance of Runge_Kutta is an iterative numerical method defined by its
    butcher array
    '''

    def __init__(self, A, b, c):
        self.stages = len(b)
        self.A = A
        self.b = b
        self.c = c
