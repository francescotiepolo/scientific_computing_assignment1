import numpy as np

# Solve the wave equation using the finite difference method taking as input Psi(x, t=0)
def solve_wave_eq(psi_0, x, t, N, c, dt, dx):
    psi_prev = np.zeros_like(x) # Psi(x, t - dt), initially 0
    psi_curr = psi_0 # Psi(x, t), initially psi_0

    output = [psi_curr.copy()] # Store results for animation

    for _ in range(t): # Loop over every timestep
        psi_next = np.zeros_like(x) # Initialize Psi(x, t + dt)

        for i in range(1, N): # Loop over every length unit
            # Update Psi(x, t + dt) using the finite difference method
            psi_next[i] = (
                2 * psi_curr[i] - psi_prev[i] +
                (c**2) * ((dt/dx)**2) * (psi_curr[i+1] - 2 * psi_curr[i] + psi_curr[i-1])
            )

        # Set boundary conditions
        psi_next[0] = 0
        psi_next[N] = 0

        # Shift Psi values for next iteration
        psi_prev, psi_curr = psi_curr, psi_next

        output.append(psi_curr.copy())

    return output