import numpy as np

def successive_over_relaxation_obj(grid, max_iters, w, p, object_mask):
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
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if object_mask[i, j] == 1:
                    new_grid[i, j] = 0.0
                else:
                    new_grid[i, j] = (w / 4) * (new_grid[i+1, j] +
                                            new_grid[i-1, j] +
                                            new_grid[i, j+1] +
                                            new_grid[i, j-1]) + (1 - w) * old_grid[i, j] 
            
        # Max difference for convergence check
        delta = np.max(np.abs(new_grid[1:-1, 1:-1] - old_grid[1:-1, 1:-1]))

        # Update the boundaries
        for j in range(cols):
            new_grid[0, j] = 0.0  # Prescribed boundary condition
            if object_mask[rows-1, j] == 1:  # Bottom boundary
                new_grid[rows-1, j] = 0.0
            else:
                new_grid[rows-1, j] = 1.0  # Prescribed boundary condition

        for i in range(rows):
            if object_mask[i, 0] == 1:  # Left boundary
                new_grid[i, 0] = 0.0
            else:
                new_grid[i, 0] = new_grid[i, 1]  # Normal update

            if object_mask[i, cols-1] == 1:  # Right boundary
                new_grid[i, cols-1] = 0.0
            else:
                new_grid[i, cols-1] = new_grid[i, cols-2]  # Normal update

        counter += 1

        # Max difference for convergence check
        delta = 0.0
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                diff = abs(new_grid[i, j] - old_grid[i, j])
                if diff > delta:
                    delta = diff
        if np.float64(delta) < np.float64(tol):
            break
  
    return new_grid, counter