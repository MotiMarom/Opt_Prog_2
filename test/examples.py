import numpy as np


def test_qp(X, t):
    # f(X)
    func_x = X[0]**2 + X[1]**2 + (X[2]+1)**2

    # Gradient
    grad_x = np.array(np.zeros(3), dtype=np.float64)
    grad_x[0] = 2*t*X[0] - 1/X[0]
    grad_x[1] = 2*t*X[1] - 1/X[1]
    grad_x[2] = 2*t*(X[2] + 1) - 1/X[2]

    # Hessian
    hessian_x = np.array(np.zeros((3, 3)), dtype=np.float64)
    hessian_x[0][0] = 2*t - 1/X[0]
    hessian_x[1][1] = 2*t - 1/X[1]
    hessian_x[2][2] = 2*t - 1/X[2]

    return func_x, grad_x, hessian_x
    # best so far: g+ h-


def test_lp(X, t):
    # f(X)
    func_x = X[0] + X[1]

    # Gradient
    grad_x = np.array(np.zeros(2), dtype=np.float64)

    grad_x[0] = t + 1/(X[0]+X[1]-1) - 1/(2-X[0])
    grad_x[1] = t + 1/(X[0]+X[1]-1) - 1/(1-X[1]) + 1/X[1]

    # Hessian
    func_h = (1 / (X[0]+X[1]-1))**2
    hessian_x = np.array(np.zeros((2, 2)), dtype=np.float64)
    hessian_x[0][0] = func_h + (1/(2-X[0]))**2
    hessian_x[0][1] = func_h
    hessian_x[1][0] = func_h
    hessian_x[1][1] = func_h + (1/(X[1]-1))**2 + (1/X[1])**2

    return func_x, grad_x, hessian_x

