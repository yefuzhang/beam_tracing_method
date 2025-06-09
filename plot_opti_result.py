import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.config import Config
from scipy.signal import convolve2d
from multiprocessing import Pool, cpu_count, freeze_support
import matplotlib.pyplot as plt
from numba import njit, prange
import os
import time


FOV_num = 6
filename = f"optimize_{FOV_num}_con_uni_0.75.npz"

with np.load(filename, allow_pickle=True) as data:
    X = data['X']
    energies = data['energy']
    uniformities = data['uniformity']

mask = uniformities >= 0.8

plt.figure(figsize=(6, 5))
plt.scatter(energies[mask], uniformities[mask], c='green', s=10, label='Feasible')
plt.scatter(energies[~mask], uniformities[~mask], c='red', s=10, label='Infeasible')
plt.axhline(0.8, color='black', linestyle='--', label='Uniformity ≥ 0.8')
plt.xlabel("Total Energy")
plt.ylabel("Uniformity")
plt.title("Pareto Front with Constraint")
plt.grid(True)
plt.legend()
plt.tight_layout()

filename = f"pareto_front_{FOV_num}.svg"
plt.savefig(filename, format='svg')

plt.show()