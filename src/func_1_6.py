import numpy as np
from numba import jit

# Due to some problems with numba handling the delta/tol comparison, we create the following functions to be optimized by numba and later called in the main functions
@jit(nopython=True)
def update_j(old_grid, new_grid, rows, cols):
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            new_grid[i, j] = 0.25 * (
                old_grid[i + 1, j] +
                old_grid[i - 1, j] +
                old_grid[i, j + 1] +
                old_grid[i, j - 1]
            )
    return new_grid

@jit(nopython=True)
def update_g(old_grid, new_grid, rows, cols):
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            new_grid[i, j] = 0.25 * (
                old_grid[i + 1, j] +
                new_grid[i - 1, j] +
                old_grid[i, j + 1] +
                new_grid[i, j - 1]
            )
    return new_grid

@jit(nopython=True)
def update_sor(old_grid, new_grid, rows, cols, w):
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            new_grid[i, j] = (w / 4) * (new_grid[i+1, j] +
                                        new_grid[i-1, j] +
                                        new_grid[i, j+1] +
                                        new_grid[i, j-1]) + (1 - w) * old_grid[i, j] 
    return new_grid

@jit(nopython=True)
def update_boundaries(grid, cols, rows):
    for j in range(cols):
        grid[0, j] = 0.0
        grid[rows - 1, j] = 1.0
    for i in range(rows):
        grid[i, 0] = grid[i, 1]
        grid[i, cols - 1] = grid[i, cols - 2]
    return grid

@jit(nopython=True)
def delta_compute(new_grid, old_grid, rows, cols):
    delta = 0.0
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            diff = abs(new_grid[i, j] - old_grid[i, j])
            if diff > delta:
                delta = diff
    return delta


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
    tol = np.float64(10**-p)
    counter = 0
    for _ in range(max_iters):
        old_grid = new_grid.copy()  # Store the previous iteration

        new_grid = update_j(old_grid, new_grid, rows, cols)

        # Now update the boundaries
        new_grid = update_boundaries(new_grid, cols, rows)

        counter += 1

        # Max difference for convergence check
        delta = delta_compute(new_grid, old_grid, rows, cols)
        if delta < tol:
            break

    return new_grid, counter

def gauss_seidel_iteration(grid, max_iters, p):
    """
    Gauss Seidel iteration solving at steady-state.

    Inputs:
        grid (numpy.ndarray): Initial grid.
        max_iters (int): Max number of iterations.
        tol (float): Convergence tolerance.

    Output:
        numpy.ndarray: final converged grid.
        int: number of iterations to convergence
    """
    rows, cols = grid.shape
    new_grid = np.copy(grid)
    tol = np.float64(10**-p)
    counter = 0
    for _ in range(max_iters):
        old_grid = new_grid.copy()  # Store the previous iteration

        # Update
        new_grid = update_g(old_grid, new_grid, rows, cols)

        # Now update the boundaries
        new_grid = update_boundaries(new_grid, cols, rows)
        
        counter += 1

        # Max difference for convergence check
        delta = delta_compute(new_grid, old_grid, rows, cols)
        if delta < tol:
            break

    return new_grid, counter

def successive_over_relaxation(grid, max_iters, w, p):
    """
    Successive Over Relaxation solving at steady-state.

    Inputs:
        grid (numpy.ndarray): Initial grid.
        max_iters (int): Max number of iterations.
        w (float): Relaxation factor (0 < w < 2).
        tol (float): Convergence tolerance.

    Output:
        numpy.ndarray: final converged grid.
        int: number of iterations to convergence
    """
    rows, cols = grid.shape
    new_grid = np.copy(grid)
    tol = np.float64(10**-p)
    counter = 0
    for _ in range(max_iters):
        old_grid = new_grid.copy()  # Store the previous iteration

        # Update
        new_grid = update_sor(old_grid, new_grid, rows, cols, w)

        # Now update the boundaries
        new_grid = update_boundaries(new_grid, cols, rows)

        counter += 1

        # Max difference for convergence check
        delta = delta_compute(new_grid, old_grid, rows, cols)
        if np.float64(delta) < np.float64(tol):
            break
  
    return new_grid, counter