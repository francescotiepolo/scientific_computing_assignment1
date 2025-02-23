import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erfc
from numba import jit

@jit(nopython=True)
def update_concentration(c, D, dx, dt, N):
    c_new = c.copy()
    
    for i in range(N):  # Apply periodic BC in x-direction
        for j in range(1, N-1):  # Skip top (j=N-1) and bottom (j=0) boundaries
            ip = (i + 1) % N  # Right neighbor (periodic)
            im = (i - 1) % N  # Left neighbor (periodic)
            
            c_new[i, j] = c[i, j] + (D * dt / dx**2) * (
                c[ip, j] + c[im, j] + c[i, j+1] + c[i, j-1] - 4 * c[i, j]
            )

    # Enforce boundary conditions
    c_new[:, 0] = 0   # Bottom boundary
    c_new[:, -1] = 1  # Top boundary

    return c_new

def analytical_solution(y, t, D, terms=10):
    sol = np.zeros_like(y)
    for i in range(terms):
        sol += erfc((1 - y + 2 * i) / (2 * np.sqrt(D * t))) - erfc((1 + y + 2 * i) / (2 * np.sqrt(D * t)))
    return sol

# Parameters
N = 50          # Grid points
D = 1.0         # Diffusion coefficient
dx = 1.0 / N    # Spacing
dt = 0.25 * dx**2 / D  # Timestep
max_steps = 3000  # Total simulation steps

# Initialize field
c = np.zeros((N, N))
c[:, -1] = 1  # Top boundary condition

y_vals = np.linspace(0, 1, N)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(7, 5))
line_num, = ax.plot([], [], 'o-', label='Numerical')
line_ana, = ax.plot([], [], '--', label='Analytical')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("y")
ax.set_ylabel("Concentration c(y)")
ax.set_title("Time-dependent Diffusion Equation")
ax.legend()
ax.grid()

# Animation function
def animate(step):
    global c
    c = update_concentration(c, D, dx, dt, N)
    mean_conc = c.mean(axis=0)
    line_num.set_data(y_vals, mean_conc)
    line_ana.set_data(y_vals, analytical_solution(y_vals, step * dt, D))
    ax.set_title(f"Time = {step * dt:.3f}")
    return line_num, line_ana

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=max_steps, interval=20, blit=True)
plt.show()
