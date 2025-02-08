import numpy as np
from numba import njit

@njit # Numba implementation
def solve_wave_eq(psi_0, x, t_steps, N, c, dt, dx):
    '''
    Solve the wave equation using the finite difference method
    - psi_0: psi(x, t=0) depending on the initial conidtion
    - x: spatial step
    - t_steps: number of time steps
    - N: number of spatial steps
    - c: constant
    - dt: time step size
    - dx: spatial step size
    '''
    psi_prev = np.zeros_like(x) # Psi(x, t - dt), initially 0
    psi_curr = psi_0.copy() # Psi(x, t), initially psi_0
    psi_next = np.zeros_like(x) # Preallocate Psi(x, t + dt)

    # Store results for animation
    output = np.empty((t_steps + 1, x.size))
    output[0, :] = psi_curr.copy()

    coeff = (c**2) * ((dt/dx)**2) # Precompute the constant in the finite difference equation

    for n in range(t_steps): # Loop over every timestep

        # Update Psi(x, t + dt) using the finite difference method (excluding boundries)
        psi_next[1:N] = (
            2 * psi_curr[1:N] - psi_prev[1:N] +
            coeff * (psi_curr[2:N+1] - 2 * psi_curr[1:N] + psi_curr[0:N-1])
        )

        psi_next[0] = psi_next[N] = 0 # Set boundary conditions

        # Shift Psi values for next iteration
        psi_prev, psi_curr, psi_next = psi_curr, psi_next, psi_prev

        output[n+1, :] = psi_curr.copy() # Store results

    return output