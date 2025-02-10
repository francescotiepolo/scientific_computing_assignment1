import numpy as np
import pytest

# Import function to test
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from solve_wave_eq_func import solve_wave_eq

def test_zero_init_cond():
    '''
    Test that if the initial condition is all zeros, the solution
    remains zero for all time steps.
    '''
    N = 3 # Number of spatial steps
    x = np.linspace(0, 3, N+1)
    psi0 = np.zeros_like(x) # Initial condition: all zeros
    c = 1.0
    dt = 1.0
    dx = 1.0
    t_steps = 5 # Run for 5 time steps

    result = solve_wave_eq(psi0, x, t_steps, N, c, dt, dx) # Run the simulation

    # Check that the output array has the expected shape
    assert result.shape == (t_steps+1, x.size)

    # Since the initial condition is zero every time-step should be zero.
    np.testing.assert_array_equal(result, np.zeros((t_steps+1, x.size)))


def test_one_step_update():
    '''
    Test that after one time step the update is as expected.
    
    We take:
        psi0 = [0, 1, 0, 0]
    and parameters:
        c = 1.0, dt = 1.0, dx = 1.0  (so that coeff = 1)
    the finite difference update (applied to the interior points 1 and 2) is:
    
        For index 1:
          psi_next[1] = 2*psi_curr[1] - psi_prev[1] + 1*(psi_curr[2] - 2*psi_curr[1] + psi_curr[0])
                       = 2*1 - 0 + (0 - 2*1 + 0) = 2 - 2 = 0

        For index 2:
          psi_next[2] = 2*psi_curr[2] - psi_prev[2] + 1*(psi_curr[3] - 2*psi_curr[2] + psi_curr[1])
                       = 2*0 - 0 + (0 - 0 + 1) = 1

        The boundaries are then set to zero.
    '''
    N = 3
    x = np.linspace(0, 3, N+1)
    psi0 = np.array([0.0, 1.0, 0.0, 0.0]) # Initial condition
    c = 1.0
    dt = 1.0
    dx = 1.0
    t_steps = 1

    # Expected result:
        # at t = 0: [0, 1, 0, 0]
        # at t = 1: [0, 0, 1, 0]
    expected = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

    # Run the simulation
    output = solve_wave_eq(psi0, x, t_steps, N, c, dt, dx)

    # Compare the output to the expected array
    np.testing.assert_array_almost_equal(output, expected)