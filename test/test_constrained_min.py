# Moti Marom
# ID 025372830

# test_constrained_min.py
import numpy as np
from constrained_min import interior_pt
from examples import test_qp
from examples import test_lp
from utils import plot_contour

# Get the requested function to analyze from user
print('Hello!')
print('Please pick a function for analysis from the following:')
print('1-sphere, 2-NA, 3-NA')
function_index = input('type a single number between [1, 2]:')
function_index = int(function_index)

if function_index == 1:
    print('You chose 1: sphere')
    func_name = 'sphere'
    func2min = test_qp
    x0 = np.array([0.1, 0.2, 0.7], dtype=np.float64)
elif function_index == 2:
        print('You chose 2: linear')
        func_name = 'linear'
        func2min = test_lp
        x0 = np.array([0.5, 0.75], dtype=np.float64)
else:
    print("You chose {} where it should be an integer number between 1-6. please rerun and try again."
          .format(function_index))

if function_index in range(1, 3):

    # Newton Descent ('newton')
    method = 'newton'
    results_newton = interior_pt(func2min, x0, m=4, t=0.5, miu=10, eps_barrier=1e-8, eps_newton=1e-8, max_iter=1000)

    if function_index != 1:
        # plot all methods tracks on top of the contour
        plot_contour(func2min, func_name, results_newton)

    print('End of {} analysis'.format(func_name))

print('End of running.')


