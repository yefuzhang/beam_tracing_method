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
Config.warnings['not_compiled'] = False


def eff_regions_vectorized(eff_saved_arr, eff):
    log_eff = np.log(eff)
    log_1_minus_eff = np.log(1 - eff)

    # eff_saved_arr shape: (N, 2 * len(eff))
    a = eff_saved_arr[:, 0::2]   # even indices
    b = eff_saved_arr[:, 1::2]   # odd indices

    logsums = np.sum(a * log_eff + b * log_1_minus_eff, axis=1)
    return np.exp(logsums)


def optimize_fun_fast(eff, output_mat, eff_saved_arr, area, aperture):
    energies = eff_regions_vectorized(eff_saved_arr, eff)

    # Weighted sum of output matrices
    output_mat2 = np.tensordot(energies, output_mat, axes=(0, 0))

    tot_energy = np.sum(energies * area) / 12.5579361

    output_mat3 = output_mat2[1:-1, 1:]
    conv_result = convolve2d(output_mat3, aperture, mode='valid')
    uniformity = np.min(conv_result) / np.max(conv_result)

    return [1 - tot_energy, 1 - uniformity]


class EffOptimization_Constrain(Problem):
    def __init__(self, output_mat, eff_saved, area, aperture):
        super().__init__(n_var=len(eff_saved[0]) // 2, n_obj=2, n_constr=1, xl=0.0, xu=1.0)
        self.output_mat = output_mat
        self.eff_saved = eff_saved
        self.area = area
        self.aperture = aperture

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = []
        f2 = []
        g1 = []  # constraint

        for eff in X:
            tot_energy, uniformity = optimize_fun_fast(
                eff, self.output_mat, self.eff_saved, self.area, self.aperture
            )
            f1.append(tot_energy)
            f2.append(uniformity)

            # Constraint: 1 - uniformity ≤ 0.25 → uniformity ≥ 0.75
            g1.append(uniformity-0.25)  # g ≤ 0 means feasible

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1])


class EffOptimization(Problem):
    def __init__(self, output_mat, eff_saved, area, aperture, mode):
        self.mode = mode
        n_var = len(eff_saved[0]) // 2

        if mode == "multi":
            super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0.0, xu=1.0)
        else:
            super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=0.0, xu=1.0)

        self.output_mat = output_mat
        self.eff_saved = eff_saved
        self.area = area
        self.aperture = aperture

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = []
        f2 = []

        for eff in X:
            tot_energy, uniformity = optimize_fun_fast(eff, output_mat, eff_saved, area, aperture)

            if self.mode == "multi":
                f1.append(tot_energy)
                f2.append(uniformity)
            else:
                f1.append(uniformity)
        
        if self.mode == "multi":
            out["F"] = np.column_stack([f1, f2])
        else:
            out["F"] = np.array(f1).reshape(-1, 1)


if __name__ == '__main__':
    freeze_support()

    for FOV_num in [1,2,3,4,5,6,7,8,9]:
        folder = f"results_{FOV_num}"
        filename = f"data_{FOV_num}_final.npz"
        path = os.path.join(folder, filename)

        # Generate human eye retina aperture
        resolution = 0.1
        x = np.arange(-2, 2 + resolution, resolution)

        y = x
        X, Y = np.meshgrid(x, y)
        circle_mask = (X**2 + Y**2) <= 2**2
        aperture = np.zeros_like(X, dtype=int)
        aperture[circle_mask] = 1

        # Loading data
        print(f"Loading {filename}...")
        with np.load(path, allow_pickle=True) as data:
            output_mat = data['output_mat']
            eff_saved = data['eff_saved']
            area = data['area']
        data = []
        start_time = time.time()

        print(f"Regions Number: {len(output_mat)}")

        # choose optimize mode: (multi, uni, ener)
        optimize_mode = "multi"

        # popsize and generation number:
        pop_size = 100
        n_gen = 1000

        # Define and run optimizer
        if optimize_mode == "multi":
            problem = EffOptimization_Constrain(output_mat, eff_saved, area, aperture)
            algorithm = NSGA2(pop_size=pop_size)
        else:
            from pymoo.algorithms.soo.nonconvex.de import DE
            problem = EffOptimization(output_mat, eff_saved, area, aperture, mode="single")
            algorithm = DE(pop_size=pop_size)
        termination = get_termination("n_gen", n_gen)

        res = minimize(problem, algorithm, termination, seed=1, verbose=True)
        end_time = time.time()
        print(f"Elapsed time (FOV: {FOV_num}): {end_time - start_time:.2f} seconds")

        if optimize_mode == "multi":
            F_raw = res.F
            energies = 1 - F_raw[:, 0]
            uniformities = 1 - F_raw[:, 1]

            filename = f"optimize_{FOV_num}_con_uni_0.75_ener"
            np.savez_compressed(filename, X=res.X, energy=energies, uniformity=uniformities, num_regions=len(area))

        else:
            best_eff = res.X
            best_uniformity = 1 - res.F

            print("Best Efficiency:", best_eff)
            print("Best Value:", best_uniformity)