import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mosek
import os
from typing import List
from src.generate import GenerateProblem


cwd = os.getcwd()
spec_files_dir = os.path.join(cwd, 'spec_files') # Contains the initial specifications
sa_files_dir = os.path.join(spec_files_dir, 'successive_augmentation')
results_dir = os.path.join(cwd, 'results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(sa_files_dir, exist_ok=True)


class SolveILP:

    def __init__(self, file, num_blocks, underestimation=True):
        self.problem = GenerateProblem(file, num_blocks, underestimation=underestimation)
        self.num_hard_modules, self.num_soft_modules = self.problem.num_hard_modules, self.problem.num_soft_modules
        self.num_total_modules = self.problem.num_total_modules
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
        self.x_ij = cp.Variable((self.num_total_modules, self.num_total_modules), integer=True)
        self.y_ij = cp.Variable((self.num_total_modules, self.num_total_modules), integer=True)

        if self.problem.soft_exists:
            self.w = cp.Variable(self.num_soft_modules)     # Soft module widths
            self.h = np.zeros(self.num_soft_modules)
        else:
            self.w, self.h = 0, 0
        self.Y = cp.Variable()

        self.objective = cp.Minimize(self.Y)
        self.constraints = []

    def create_constraints(self):

        # Hard-Hard Non-overlap #

        if self.problem.hard_exists:
            for i in range(self.num_hard_modules):
                for j in range(self.num_hard_modules):
                    if j > i:
                        self.constraints.append(self.x[i] + self.z[i] * self.hard_module_height[i] + (1-self.z[i]) * self.hard_module_width[i] <= self.x[j] + self.bound * (self.x_ij[i, j] + self.y_ij[i, j]))
                        self.constraints.append(self.x[i] - self.z[j] * self.hard_module_height[j] - (1-self.z[j]) * self.hard_module_width[j] >= self.x[j] - self.bound * (1 - self.x_ij[i, j] + self.y_ij[i, j]))
                        self.constraints.append(self.y[i] + self.z[i] * self.hard_module_width[i] + (1-self.z[i]) * self.hard_module_height[i] <= self.y[j] + self.bound * (1 + self.x_ij[i, j] - self.y_ij[i, j]))
                        self.constraints.append(self.y[i] - self.z[j] * self.hard_module_width[j] - (1-self.z[j]) * self.hard_module_height[j] >= self.y[j] - self.bound * (2 - self.x_ij[i, j] - self.y_ij[i, j]))

        # Hard-Soft Non-overlap #

        if self.problem.hard_exists and self.problem.soft_exists:
            for i in range(self.num_hard_modules):
                for j in range(self.num_hard_modules, self.num_total_modules):
                    if j > i:
                        self.constraints.append(self.x[i] + self.z[i] * self.hard_module_height[i] + (1-self.z[i]) * self.hard_module_width[i] <= self.x[j] + self.bound * (self.x_ij[i, j] + self.y_ij[i, j]))
                        self.constraints.append(self.x[i] - self.w[j-self.num_hard_modules] >= self.x[j] - self.bound * (1 - self.x_ij[i, j] + self.y_ij[i, j]))
                        self.constraints.append(self.y[i] + self.z[i] * self.hard_module_width[i] + (1-self.z[i]) * self.hard_module_height[i] <= self.y[j] + self.bound * (1 + self.x_ij[i, j] - self.y_ij[i, j]))
                        self.constraints.append(self.y[i] - (self.gradient[j-self.num_hard_modules] * self.w[j-self.num_hard_modules] + self.intercept[j-self.num_hard_modules]) >= self.y[j] - self.bound * (2 - self.x_ij[i, j] - self.y_ij[i, j]))

        # Soft-Soft Non-overlap #

        if self.problem.soft_exists:
            for i in range(self.num_hard_modules, self.num_total_modules):
                for j in range(self.num_hard_modules, self.num_total_modules):
                    if j > i:
                        self.constraints.append(self.x[i] + self.w[i-self.num_hard_modules] <= self.x[j] + self.bound * (self.x_ij[i, j] + self.y_ij[i, j]))
                        self.constraints.append(self.x[i] - self.w[j-self.num_hard_modules] >= self.x[j] - self.bound * (1 - self.x_ij[i, j] + self.y_ij[i, j]))
                        self.constraints.append(self.y[i] + (self.gradient[i-self.num_hard_modules] * self.w[i-self.num_hard_modules] + self.intercept[i-self.num_hard_modules]) <= self.y[j] + self.bound * (1 + self.x_ij[i, j] - self.y_ij[i, j]))
                        self.constraints.append(self.y[i] - (self.gradient[j-self.num_hard_modules] * self.w[j-self.num_hard_modules] + self.intercept[j-self.num_hard_modules]) >= self.y[j] - self.bound * (2 - self.x_ij[i, j] - self.y_ij[i, j]))

        for x in self.x:
            self.constraints.append(x >= 0)
        for y in self.y:
            self.constraints.append(y >= 0)

        for i in range(self.num_soft_modules):
            w_min = self.soft_module_width_range[i, 0]
            w_max = self.soft_module_width_range[i, 1]

            self.constraints.append(w_min <= self.w[i])
            self.constraints.append(self.w[i] <= w_max)

        for i in range(self.num_total_modules):
            for j in range(self.num_total_modules):
                if j > i:
                    self.constraints.append(0 <= self.x_ij[i, j])
                    self.constraints.append(self.x_ij[i, j] <= 1)
                    self.constraints.append(0 <= self.y_ij[i, j])
                    self.constraints.append(self.y_ij[i, j] <= 1)

        if self.problem.hard_exists:
            for z in self.z:
                self.constraints.append(0 <= z)
                self.constraints.append(z <= 1)

            for i in range(self.num_hard_modules):
                self.constraints.append(self.x[i] + self.z[i] * self.hard_module_height[i] + (1-self.z[i]) * self.hard_module_width[i] <= self.Y)
                self.constraints.append(self.y[i] + self.z[i] * self.hard_module_width[i] + (1-self.z[i]) * self.hard_module_height[i] <= self.Y)

        if self.problem.soft_exists:
            for i in range(self.num_hard_modules, self.num_total_modules):
                self.constraints.append(self.x[i] + self.w[i-self.num_hard_modules] <= self.Y)
                self.constraints.append(self.y[i] + (self.gradient[i-self.num_hard_modules] * self.w[i-self.num_hard_modules] + self.intercept[i-self.num_hard_modules]) <= self.Y)

        return self.constraints

    def solve(self, run_time, solver='MOSEK', verbose=False):
        model = cp.Problem(self.objective, self.constraints)
        model.solve(solver=solver, verbose=verbose, mosek_params={mosek.dparam.optimizer_max_time: run_time})
        if self.problem.hard_exists and self.problem.soft_exists:
            for i in range(self.num_soft_modules):
                self.h[i] = self.gradient[i] * self.w.value[i] + self.intercept[i]

            return model.value, self.x.value, self.y.value, self.z.value, self.w.value, self.h
        elif self.problem.hard_exists and not self.problem.soft_exists:

            return model.value, self.x.value, self.y.value, self.z.value, self.w, self.h

        elif not self.problem.hard_exists and self.problem.soft_exists:
            for i in range(self.num_soft_modules):
                self.h[i] = self.gradient[i] * self.w.value[i] + self.intercept[i]

        return model.value, self.x.value, self.y.value, self.z, self.w.value, self.h    # W and H are soft module widths and heights

    def visualize(self, bound, X, Y, Z, W, H, idx=1, glob=False, sa=True, show_layout=True, utilizations=[1]): # W and H are soft module widths and heights
        if self.problem.hard_exists and self.problem.soft_exists:
            W = np.concatenate((self.hard_module_width, W))
            H = np.concatenate((self.hard_module_height, H))
        elif self.problem.hard_exists and not self.problem.soft_exists:
            W = self.hard_module_width
            H = self.hard_module_height

        chip_area = bound ** 2
        self.utilization = (np.sum(W * H) / chip_area) * np.prod(utilizations)

        label = np.arange(self.num_total_modules) + 1
        plt.ion()
        fig, ax = plt.subplots()
        for i, txt in enumerate(label):
            if i < self.num_hard_modules:
                if Z[i] >= 0.9: # Sometimes get 1.01/0.99
                    ax.add_patch(Rectangle((X[i], Y[i]), H[i], W[i], color='red'))
                    ax.add_patch(Rectangle((X[i], Y[i]), H[i], W[i], color='black', fill=False))
                    ax.annotate(text=txt, xy=(X[i], Y[i]), xytext=(X[i]+H[i]/2, Y[i]+W[i]/2))
                else:
                    ax.add_patch(Rectangle((X[i], Y[i]), W[i], H[i], color='green'))
                    ax.add_patch(Rectangle((X[i], Y[i]), W[i], H[i], color='black', fill=False))
                    ax.annotate(text=txt, xy=(X[i], Y[i]), xytext=(X[i]+W[i]/2, Y[i]+H[i]/2))
            else:
                ax.add_patch(Rectangle((X[i], Y[i]), W[i], H[i], color='yellow'))
                ax.add_patch(Rectangle((X[i], Y[i]), W[i], H[i], color='black', fill=False))
                ax.annotate(text=txt, xy=(X[i], Y[i]), xytext=(X[i]+W[i]/2, Y[i]+H[i]/2))
            if sa==True:
                if glob==False:
                        plt.title('Local floorplan for %d-th sub-block: Chip Height = %.4f, Chip Area = %d\nUtilization = %.2f percent' % (idx, bound, chip_area, self.utilization * 100))
                else:
                        plt.title('Global floorplan for including all sub-blocks: Chip Height = %.4f, Chip Area = %d\nUtilization = %.2f percent' % (bound, chip_area, self.utilization * 100))
            else:
                plt.title('Direct floorplan: Chip Height = %.4f, Chip Area = %d\nUtilization = %.2f percent' % (bound, chip_area, self.utilization * 100))

        ax.set_xlim(0, bound)
        ax.set_ylim(0, bound)
        if show_layout:
            plt.show(block=True)
        else:
            plt.close()
        return W, H

    def save_augmented_dimensions(self, num_blocks:int, bounds):
        """
            args:
                bounds - the list of bounds for every superblock
        """
        f = open(os.path.join(sa_files_dir, f'{num_blocks}', f'{num_blocks}_blocks_sa.ilp'), 'w')
        f.write(f'hard - {len(bounds)}\n')
        for bound in bounds:
            f.write(f'{bound},{bound}\n')
        f.close()

    def save_final_dimensions(self, bound, num_blocks, sa=True):
        res_file_name = f'{num_blocks}_sa_{sa}_dimensions.txt'
        res_file_path = os.path.join(results_dir, res_file_name)
        f = open(res_file_path, 'w')
        f.write(f'{bound},{bound}\n')
        f.close()