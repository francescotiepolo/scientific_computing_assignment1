import numpy as np

def jacobi_iteration(grid, max_iters, p):
    """
    Jacobi iteration solving at steady-state.

    Inputs:
        grid (numpy.ndarray): Initial grid.
        max_iters (int): Max number of iterations.
        tol (float): Convergence tolerance.

    Output:
        numpy.ndarray: The final converged grid.
        int: number of iterations to convergence
    """
    rows, cols = grid.shape
    new_grid = np.copy(grid)  
    tol = 10**-p
    counter = 0
    for _ in range(max_iters):
        old_grid = new_grid.copy()  # Store the previous iteration

        # Update
        for i in range(0, rows):
            for j in range(0, cols):
                new_grid[i, j] = (1/4) * (old_grid[i + 1, j] + old_grid[i - 1, j] + old_grid[i, j + 1] + old_grid[i, j - 1])

        new_grid[:, 0] = new_grid[:, -1] # Boundry conditions
        # Reapply fixed y-boundaries (they remain unchanged)
        new_grid[0, :] = 0 # bottom: c = 0
        new_grid[-1, :] = 1 # top: c = 1

        # Max difference for convergence check
        delta = np.max(np.abs(new_grid - old_grid))
        counter += 1
        if delta < tol:
            break

    return new_grid, counter