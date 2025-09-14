import numpy as np
import cvxpy as cp
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mosek
import os
from typing import List
from src.generate.generate import GenerateProblem
from src.solve.hard_hard_nonoverlap import hard_hard_nonoverlap
from src.solve.hard_soft_nonoverlap import hard_soft_nonoverlap
from src.solve.soft_soft_nonoverlap import soft_soft_nonoverlap
from src.solve.other_constraints import other_constraints


cwd = os.getcwd()
spec_files_dir = os.path.join(cwd, 'spec_files') # Contains the initial specifications
sa_files_dir = os.path.join(spec_files_dir, 'successive_augmentation')
results_dir = os.path.join(cwd, 'results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(sa_files_dir, exist_ok=True)


class SolveILP:

    def __init__(self, file, underestimation=True, save_lp=True):
        self.problem = GenerateProblem(file, underestimation=underestimation, save_lp=save_lp)
        self.num_hard_modules, self.num_soft_modules = self.problem.num_hard_modules, self.problem.num_soft_modules
        self.num_total_modules = self.problem.num_total_modules
        self.hard_exists, self.soft_exists = self.problem.hard_exists, self.problem.soft_exists
        self.hard_module_width, self.hard_module_height = self.problem.hard_module_width, self.problem.hard_module_height
        self.soft_module_width_range, self.soft_module_height_range = self.problem.soft_module_width_range, self.problem.soft_module_height_range
        self.soft_area = self.problem.area
        self.gradient, self.intercept = self.problem.gradient, self.problem.intercept
        self.bound = self.problem.bound

        self.x = cp.Variable(self.num_total_modules)
        self.y = cp.Variable(self.num_total_modules)

        if self.problem.hard_exists:
            self.z = cp.Variable(self.num_hard_modules, integer=True)
        else:
            self.z = 0
        self.x_ij = cp.Variable((self.num_total_modules,
                                 self.num_total_modules), integer=True)
        self.y_ij = cp.Variable((self.num_total_modules,
                                 self.num_total_modules), integer=True)

        if self.problem.soft_exists:
            self.w = cp.Variable(self.num_soft_modules)     # Soft module widths
            self.h = np.zeros(self.num_soft_modules)
        else:
            self.w, self.h = 0, 0
        self.Y = cp.Variable()

        self.objective = cp.Minimize(self.Y)
        self.constraints = []

    def create_constraints(self):

        hard_hard_nonoverlap(self.hard_exists,
                             self.num_hard_modules,
                             self.hard_module_width,
                             self.hard_module_height,
                             self.bound,
                             self.x, self.y, self.x_ij,
                             self.y_ij, self.z,
                             self.constraints)
        
        hard_soft_nonoverlap(self.hard_exists,
                             self.soft_exists,
                             self.num_hard_modules,
                             self.num_total_modules,
                             self.hard_module_width,
                             self.hard_module_height,
                             self.gradient, self.intercept,
                             self.bound,
                             self.x, self.y, self.x_ij,
                             self.y_ij, self.z, self.w,
                             self.constraints)

        soft_soft_nonoverlap(self.soft_exists,
                             self.num_hard_modules,
                             self.num_total_modules,
                             self.gradient, self.intercept,
                             self.bound,
                             self.x, self.y, self.x_ij,
                             self.y_ij, self.w,
                             self.constraints)

        other_constraints(self.hard_exists,
                          self.soft_exists,
                          self.num_hard_modules,
                          self.num_soft_modules,
                          self.num_total_modules,
                          self.hard_module_width,
                          self.hard_module_height,
                          self.gradient, self.intercept,
                          self.soft_module_width_range,
                          self.x, self.y, self.x_ij, self.y_ij,
                          self.z, self.w, self.Y,
                          self.constraints)
        
        return self.constraints

    def solve(self, run_time, solver='MOSEK', verbose=False):
        model = cp.Problem(self.objective, self.constraints)
        model.solve(solver=solver, verbose=verbose, mosek_params={mosek.dparam.optimizer_max_time: run_time})

        W, H = np.zeros(self.num_total_modules), np.zeros(self.num_total_modules)
        if self.problem.hard_exists:
            self.z = self.z.value
            rotate_index = self.z > 0.5
            W[:self.num_hard_modules] = self.hard_module_width
            W[:self.num_hard_modules][rotate_index] = self.hard_module_height[rotate_index]

            H[:self.num_hard_modules] = self.hard_module_height
            H[:self.num_hard_modules][rotate_index] = self.hard_module_width[rotate_index]
        if self.problem.soft_exists:
            self.w = self.w.value
            for i in range(self.num_soft_modules):
                self.h[i] = self.gradient[i] * self.w[i] + self.intercept[i]
            W[self.num_hard_modules:self.num_total_modules] = self.w
            H[self.num_hard_modules:self.num_total_modules] = self.h
        
        X, Y, Z = self.x.value, self.y.value, self.z

        # Shift modules if doesn't start from 0
        X -= X.min()
        Y -= Y.min()

        # Get corrected chip dimensions
        chip_height = (Y + H).max()
        chip_width = (X + W).max()

        return chip_height, chip_width, X, Y, Z, W, H # W and H are soft module widths and heights
    
    def compute_utilization(self, chip_height, chip_width, H, W, utilizations=[1]):
        chip_area = chip_height * chip_width
        self.utilization = (np.sum(W * H * utilizations) / chip_area)

    def visualize(self, chip_height, chip_width, X, Y, Z, W, H, idx=1,
                  glob=False, sa=True,
                  show_layout=True):
        chip_area = chip_height * chip_width

        label = np.arange(self.num_total_modules) + 1
        plt.ion()
        fig, ax = plt.subplots()
        for i, txt in enumerate(label):
            if i < self.num_hard_modules:
                if Z[i] >= 0.9: # Sometimes get 1.01/0.99
                    ax.add_patch(Rectangle((X[i], Y[i]), W[i], H[i], color='red'))
                else:
                    ax.add_patch(Rectangle((X[i], Y[i]), W[i], H[i], color='green'))
            else:
                ax.add_patch(Rectangle((X[i], Y[i]), W[i], H[i], color='yellow'))
            ax.add_patch(Rectangle((X[i], Y[i]), W[i], H[i], color='black', fill=False))
            ax.annotate(text=txt, xy=(X[i], Y[i]), xytext=(X[i]+W[i]/2, Y[i]+H[i]/2))
            if sa==True:
                if glob==False:
                    plt.title('Local floorplan for %d-th sub-block: Chip Height = %.4f, Chip Area = %d\nUtilization = %.2f percent' % (idx, chip_height, chip_area, self.utilization * 100))
                else:
                    plt.title('Global floorplan for including all sub-blocks: Chip Height = %.4f, Chip Area = %d\nUtilization = %.2f percent' % (chip_height, chip_area, self.utilization * 100))
            else:
                plt.title('Direct floorplan: Chip Height = %.4f, Chip Area = %d\nUtilization = %.2f percent' % (chip_height, chip_area, self.utilization * 100))

        ax.set_xlim(0, chip_width)
        ax.set_ylim(0, chip_height)
        if show_layout:
            plt.show(block=sa and glob or not sa)
        else:
            plt.close()
        return W, H

def save_augmented_dimensions(num_blocks:int, chip_heights, chip_widths):
    """
        args:
            bounds - the list of bounds for every superblock
    """
    f = open(os.path.join(sa_files_dir, f'{num_blocks}', f'{num_blocks}_blocks_sa.ilp'), 'w')
    f.write(f'hard - {len(chip_heights)}\n')
    for chip_height, chip_width in zip(chip_heights, chip_widths):
        f.write(f'{chip_width},{chip_height}\n')
    f.close()

def save_final_dimensions(chip_height, chip_width, num_blocks, sa=True):
    res_file_name = f'{num_blocks}_sa_{sa}_dimensions.txt'
    res_file_path = os.path.join(results_dir, res_file_name)
    f = open(res_file_path, 'w')
    f.write(f'{chip_width},{chip_height}\n')
    f.close()