# Moti Marom
# ID 025372830

# constrained_min.py

import numpy as np


def wolfe_backtrack(func2min, xi, step_direct, t):
    """
    Wolfe backtracking condition.

    Input:
    - func2min: objective function to minimize
    - x: current position.
    - p: step direction

    Output:
    - alpha: i.e., step length

    """
    # Init
    # make sure 0 < c1 < c2 < 1
    alpha = 1.0
    c_wolfe_1 = 0.0001
    c_wolfe_2 = 0.9  # not needed
    alpha_factor = 0.25
    hessian_flag = False
    max_iter = 1000
    alpha_ok = True

    # current position
    p = step_direct
    # func_x, grad_x, NA = func2min(xi, hessian_flag)
    func_x, grad_x, NA = func2min(xi, t)
    grad_proj_step = np.dot(grad_x, p)

    # next position
    x_next = xi - alpha * p
    # func_x_next, grad_x_next, NA = func2min(x_next, hessian_flag)
    func_x_next, grad_x_next, NA = func2min(x_next, t)
    # wolfe #1 threshold
    d_grad_proj_step = alpha * c_wolfe_1 * grad_proj_step
    # wolfe #2 threshold
    # next_step_grad_proj = np.dot(grad_x_next, p)

    i_iter = 0
    # while (((func_x_next > func_x + d_grad_proj_step) or # wolfe 1
    #        (next_step_grad_proj < c_wolfe_2 * grad_proj_step)) and # wolfe 2
    #       (i_iter <= max_iter)):
    while ((func_x_next > func_x - d_grad_proj_step) and  # wolfe 1
           (i_iter <= max_iter)):
        # search for alpha
        i_iter += 1
        alpha *= alpha_factor
        x_next = xi - alpha * p
        func_x_next, grad_x_next, NA = func2min(x_next, t)
        # func_x_next, grad_x_next, NA = func2min(x_next, hessian_flag)
        d_grad_proj_step = alpha * c_wolfe_1 * grad_proj_step
        # print(" Backtrack loop({}): f(x) = {},  f(x+ap) = {} :"
        #      .format(i_iter, func_x, func_x_next))

    #print(" Backtrack: alpha = {}, f(x) = {},  f(x+ap) = {}, dgrad(x) = {} :"
    #      .format(alpha, func_x, func_x_next, d_grad_proj_step))

    if i_iter > max_iter:
        alpha_ok = False

    return alpha, alpha_ok


def interior_pt(func2min, x0, m, t, miu=10, eps_barrier=1e-8, eps_newton=1e-8, max_iter=1000):

    """
    Barrier method algorithm.

    Input:
    - func2min: objective function to minimize
    - x0: initial position.
    - m: number of constraints
    - t: user's parameter
    - miu: t slope
    - eps_barrier: barrier stop criteria
    - eps_newton: newton step stop criteria
    - max_iter: inner loop max number of iterations

    Output: track records
    - x_track, f_track & barrier_factor

    """

    xi = np.copy(x0)  # store initial value
    x_track = []  # keep track on x
    f_track = []  # keep track on f(x)
    barrier_factor = []  # keep track on barrier_factor = m/t
    outer_iter = 0  # number of iterations of outer loop

    # loop until stopping criterion is met
    while m/t > eps_barrier:
        # Get f(x), g(x) & H(x)
        func_x, grad_x, hessian_x = func2min(xi, t)

        # append to track records
        x_track.append(xi)
        f_track.append(func_x)
        barrier_factor.append(m/t)

        # find Newton direction using equation solver:
        newton_step = np.linalg.solve(hessian_x, grad_x)

        # centering step: Newton Algorithm
        inner_iter = 0

        while inner_iter < max_iter:
            prev_func_x = np.copy(func_x)
            # find step length using wolfe backtrack condition
            alpha, alpha_ok = wolfe_backtrack(func2min, xi, newton_step, t)

            if alpha_ok == False:
                break

            # take 1 step towards opposite of current gradient
            xi -= alpha * newton_step

            # set the next step direction
            func_x, grad_x, hessian_x = func2min(xi, t)

            df = prev_func_x - func_x
            if df <= eps_newton:
                #print("Newton termination: small df =", df)
                break

            newton_step = np.linalg.solve(hessian_x, grad_x)

            inner_iter += 1

        # print result
        print('Barrier iteration #{}: x = {}, f(x) = {}, m/t = {}'.format(outer_iter, xi, func_x, m/t))

        # update parameter t
        t *= miu

        # append to track records
        x_track.append(xi)
        f_track.append(func_x)
        barrier_factor.append(m/t)
        outer_iter += 1


    # convert lists to numpy arrays
    x_track = np.array(x_track)
    f_track = np.array(f_track)
    barrier_factor = np.array(barrier_factor)
    final_x = xi
    final_fx = func_x

    return final_x, final_fx, x_track, f_track


def interior_pt_old(func2min, x0, m, t, miu=10, eps_barrier=1e-5, eps_newton=1e-5, max_iter=1000):

    """
    Barrier method algorithm.

    Input:
    - func2min: objective function to minimize
    - x: current position.
    - p: step direction

    Output:
    - alpha: i.e., step length

    """

    xi = x0  # store initial value
    x_track = [xi]  # keep track on x
    func_x, grad_x, hessian_x = func2min(xi, t)
    f_track = [func_x]  # keep track on f(x)
    barrier_factor = [m / t]  # keep track on barrier_factor = m/t
    k = 0  # number of iterations
    print('Barrier start: x = {}, f(x) = {}'.format(xi, f_track[k]))

    # loop until stopping criterion is met
    while m / t > eps_barrier:
        # centering step: Newton Algorithm
        i = 0
        d = np.array([[1], [1]])
        while np.linalg.norm(newton_step) > eps_newton and i < max_iter:
            func_x, grad_x, hessian_x = func2min(xi, t)
            # append to track records
            x_track.append(xi)
            f_track.append(func_x)
            # find Newton direction using equation solver:
            newton_step = np.linalg.solve(hessian_x, grad_x)
            # find step length using wolfe backtrack condition
            alpha, alpha_ok = wolfe_backtrack(func2min, xi, newton_step)

            if alpha_ok == False:
                break

            # take 1 step towards opposite of current gradient
            xi -= alpha * newton_step

            i += 1

        # update parameter t
        t *= miu

        # update tabulations
        barrier_factor.append(m / t)
        k += 1

        # print result
        print('Barrier iteration #{}: x = {}, f(x) = {}, m/t = {}'.format(k, xi, f_track[k], barrier_factor[k]))

    x_track = np.array(x_track)
    f_track = np.array(f_track)
    barrier_factor = np.array(barrier_factor)

    return x_track, f_track, barrier_factor
