# Moti Marom
# ID 025372830

# utils.py

import numpy as np
import matplotlib.pyplot as plt


def plot_contour(obj_func, func_name, results_newton):

    # prints
    # Meshgrid between -10 and 10
    x0 = np.linspace(-10., 10., 100)
    x1 = np.linspace(-10., 10., 100)
    x2 = np.linspace(-10., 10., 100)
    mesh_x0, mesh_x1 = np.meshgrid(x0, x1)
    n_rows, n_cols = mesh_x0.shape
    func_mesh_x = np.array(np.zeros((n_rows, n_cols)), dtype=np.float64)

    for r in range(n_rows):
        for c in range(n_cols):
            x_in_to_obj_f = np.array([mesh_x0[r, c], mesh_x1[r, c]], dtype=np.float64)
            func_x, g_NA, h_NA = obj_func(x_in_to_obj_f, t=0)
            func_mesh_x[r, c] = func_x

    # Plot
    #fig = plt.figure(figsize=(20, 20))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(mesh_x0, mesh_x1, func_mesh_x, 60, cmap='viridis')


    ax.scatter(results_newton[0][0], results_newton[0][1], results_newton[1], marker='o', color='red', linewidth=10)
    ax.plot(results_newton[2][:, 0], results_newton[2][:, 1], results_newton[3], color='green', linewidth=3)


    ax.set_title('$f(x) = f$({}) track using all minimization method'.format(func_name))
    ax.set_xlabel('$x_{0}$')
    ax.set_ylabel('$x_{1}$')
    ax.set_zlabel('$f(x) = f$({})'.format(func_name))
    ax.view_init(40, 20)

    plt.legend()
    plt.show()

