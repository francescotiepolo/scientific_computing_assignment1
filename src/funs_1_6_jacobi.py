import numpy as np

def jacobi_iteration(grid, max_iters, tol=1e-5):
    """
    Jacobi iteration solving at steady-state.

    Input<:
        grid (numpy.ndarray): Initial grid.
        max_iters (int): Max number of iterations.
        tol (float): Convergence tolerance.

    Output:
        numpy.ndarray: The final converged grid.
    """
    rows, cols = grid.shape
    new_grid = np.copy(grid)  

    for _ in range(max_iters):
        old_grid = new_grid.copy()  # Store the previous iteration

        # Update excluding boundaries
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                new_grid[i, j] = (1/4) * (old_grid[i + 1, j] + old_grid[i - 1, j] + old_grid[i, j + 1] + old_grid[i, j - 1])

        # Max difference for convergence check
        delta = np.max(np.abs(new_grid - old_grid))
        if delta < tol:
            break

    return new_grid

