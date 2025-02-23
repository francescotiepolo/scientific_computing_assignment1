# scientific_computing

The first section of the assignment delves into the discretization of the wave equation through the use of a central difference scheme to simulate the motion of a vibrating string under different initial conditions. The second section explores the diffusion equation of a two-dimensional domain, comparing numerical and analytical results, and analyzing the efficiency of iterative methods such as Jacobi, Gauss-Seidel and Successive Over Relaxation (SOR).

The functions udes in the Jupiter Notebooks are stored in the src folder. The resulting figures and animations are saved in the fig folder. The Jupiter notebooks titles refer to the 3 blocks of questions in the assignment PDF. plots_1_K.ipynb refers to the last questions concerning the sinks. Test for the function used in plots_1_1 is saved in tests folder.

libraries used:
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib import cm
from matplotlib import colors
from src.solve_wave_eq_func import solve_wave_eq
from numba import jit
from scipy.special import erfc
import pytest