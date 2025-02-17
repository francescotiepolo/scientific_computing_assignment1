import numpy as np

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
    tol = 10**-p
    counter = 0
    for _ in range(max_iters):
        max_change = 0  

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                old_val = new_grid[i, j]
                
                new_val = (w / 4) * (new_grid[i+1, j] + new_grid[i-1, j] + new_grid[i, j+1] + new_grid[i, j-1]) + (1 - w) * old_val 
                
                new_grid[i, j] = new_val
                max_change = max(max_change, abs(new_val - old_val))  
                
        counter += 1 
        if max_change < tol:
            break  

    return new_grid, counter