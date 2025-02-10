import numpy as np


def gauss_seidel_iteration(grid, max_iters, tol=1e-5):
    """
    Gauss Seidel iteration solving at steady-state.

    Inputs:
        grid (numpy.ndarray): Initial grid.
        max_iters (int): Max number of iterations.
        tol (float): Convergence tolerance.

    Output:
        numpy.ndarray: final converged grid.
    """
    rows, cols = grid.shape
    new_grid = np.copy(grid)  

    for _ in range(max_iters):
        max_change = 0  # Track max change for convergence

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                old_value = new_grid[i, j]
                new_grid[i, j] = (1/4) * (new_grid[i+1, j] + new_grid[i-1, j] + new_grid[i, j+1] + new_grid[i, j-1])
                max_change = max(max_change, abs(new_grid[i, j] - old_value))  # Track largest update

        if max_change < tol:
            break

    return new_grid