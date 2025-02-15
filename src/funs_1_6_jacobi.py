import numpy as np

def jacobi_iteration(grid, max_iters, p):
    """
    Jacobi iteration solving at steady-state.

    Input<:
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

        # Update excluding boundaries
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                new_grid[i, j] = (1/4) * (old_grid[i + 1, j] + old_grid[i - 1, j] + old_grid[i, j + 1] + old_grid[i, j - 1])

        # Max difference for convergence check
        delta = np.max(np.abs(new_grid - old_grid))
        counter += 1
        if delta < tol:
            break

    return new_grid, counter

