import numpy as np
import os
from typing import List


cwd = os.getcwd()
spec_files_dir = os.path.join(cwd, 'spec_files') # Contains the initial specifications
lp_solve_files_dir = os.path.join(cwd, 'lp_solve_files') # Contains the sources files for LPSolve tool
os.makedirs(lp_solve_files_dir, exist_ok=True)


class GenerateProblem:

    def __init__(self, file, num_blocks, underestimation=True):
        """
        args:
            file: The provided *.ilp file (str)
            num_blocks: Number of blocks to be optimized (int)
            underestimation: Whether or not we are considering underestimation (bool)
        """
        self.hard_exists = False
        self.soft_exists = False
        self.underestimation = underestimation

        self.num_blocks = num_blocks
        spec_file = os.path.join(spec_files_dir, file)
        lines = []
        with open(file) as f:
            for line in f:
                lines.append(line)
        self.lines = lines
        
        self.num_hard_modules, self.num_soft_modules = self.total_modules()
        self.num_total_modules = self.num_hard_modules + self.num_soft_modules
        self.hard_module_width, self.hard_module_height = self.hard_module_dimension()
        self.area, self.min_aspect, self.max_aspect = self.soft_module_properties()
        self.soft_module_width_range, self.soft_module_height_range = self.soft_module_dimension_range()
        self.gradient, self.intercept = self.linear_approximation()
        self.bound = self.upper_bound()
        self.output = os.path.join(lp_solve_files_dir, f'{self.num_total_modules}_blocks_constraints.lp')


    def total_modules(self):
        for line in self.lines:
            if line[0:4] == 'hard':
                self.hard_exists = True
                num_hard_modules = line.replace('hard - ', '')
                num_hard_modules = int(num_hard_modules.replace('\n', ''))
            elif line[0:4] == 'soft':
                self.soft_exists = True
                num_soft_modules = line.replace('soft - ', '')
                num_soft_modules = int(num_soft_modules.replace('\n', ''))
            elif self.hard_exists and not self.soft_exists:
                num_soft_modules = 0
            elif not self.hard_exists and self.soft_exists:
                num_hard_modules = 0
            elif not self.hard_exists and not self.soft_exists:

                raise FileNotFoundError('Empty File!')

        return num_hard_modules, num_soft_modules

    def hard_module_dimension(self):
        if self.hard_exists:
            hard_module_width = []
            hard_module_height = []
            for line_count in range(len(self.lines)):
                if self.lines[line_count][0:4] == 'hard':
                    break
            for i in range (line_count+1, line_count+1+self.num_hard_modules):
                comma_ind = self.lines[i].index(',')
                width = self.lines[i][:comma_ind]
                height = self.lines[i][comma_ind+1:]
                hard_module_width.append(float(width))
                hard_module_height.append(float(height))

            hard_module_width = np.array(hard_module_width)
            hard_module_height = np.array(hard_module_height)
        else:
            hard_module_width, hard_module_height = 0, 0

        return hard_module_width, hard_module_height

    def soft_module_properties(self):
        if self.soft_exists:
            area, min_aspect, max_aspect = [], [], []

            for line_count in range(len(self.lines)):
                if self.lines[line_count][0:4] == 'soft':
                    break
            for i in range(line_count+1, line_count+1+self.num_soft_modules):
                first_comma_ind = self.lines[i].index(',')
                module_area = (self.lines[i][:first_comma_ind])
                line_after_comma = self.lines[i][first_comma_ind+1:]
                second_comma_ind = line_after_comma.index(',')
                area.append(float(module_area))
                module_min_aspect = line_after_comma[:second_comma_ind]
                module_max_aspect = line_after_comma[second_comma_ind+1:]
                min_aspect.append(float(module_min_aspect))
                max_aspect.append(float(module_max_aspect))

            area = np.array(area)
            min_aspect = np.array(min_aspect)
            max_aspect = np.array(max_aspect)
        else:
            area, min_aspect, max_aspect = 0, 0, 0

        return area, min_aspect, max_aspect

    def soft_module_dimension_range(self):
        if self.soft_exists:
            min_w = np.sqrt(self.area * self.min_aspect)[:, np.newaxis]
            max_w = np.sqrt(self.area * self.max_aspect)[:, np.newaxis]
            area = self.area[:, np.newaxis]

            soft_module_width_range = np.concatenate((min_w, max_w), axis=1)
            if self.underestimation:
                soft_module_height_range = area / max_w + (max_w - soft_module_width_range) * area / max_w ** 2
            else:
                soft_module_height_range = area / soft_module_width_range

            return soft_module_width_range, soft_module_height_range[:, ::-1]
        else:
            soft_module_width_range = np.array([[0, 0]])
            soft_module_height_range = np.array([[0, 0]])

            return soft_module_width_range, soft_module_height_range

    def linear_approximation(self):
        if self.soft_exists:
            area = self.area
            min_w = self.soft_module_width_range[:, 0]
            max_w = self.soft_module_width_range[:, 1]
            min_h = self.soft_module_height_range[:, 0]
            max_h = self.soft_module_height_range[:, 1]

            if self.underestimation:
                gradient = - area / max_w ** 2
                intercept = 2 * area / max_w
            else:
                gradient = (max_h - min_h) / (min_w - max_w)
                intercept = max_h - gradient * min_w

            return gradient, intercept
        else:

            return 0, 0

    def soft_module_height(self, w):
        if self.soft_exists:

            return w * self.gradient + self.intercept
        else:

            return 0

    def actual_soft_height(self, W):
        if self.soft_exists:

            return self.area / W
        else:

            return 0

    def upper_bound(self):
        W_hard = np.maximum(self.hard_module_width, self.hard_module_height).sum()
        H_hard = W_hard
        W_soft = self.soft_module_width_range[:, 1].sum()
        H_soft = self.soft_module_height_range[:, 1].sum()

        W = W_hard + W_soft
        H = H_hard + H_soft

        return np.max([W, H])

    def objective(self):
        g = open(self.output, 'w')
        g.write('/* Objective Function */\nmin: Y;\n\n\n')
        g.close()

    def hard_hard_nonoverlap(self):
        if self.hard_exists:
            width, height = self.hard_module_width, self.hard_module_height
            bound = self.bound
            g = open(self.output, 'a')
            g.write('/* Non-overlap constraints hard-hard */\n')
            for i in range(1, self.num_hard_modules+1):
                for j in range(1, self.num_hard_modules+1):
                    if j <= i:
                        continue
                    else:
                        g.write(f'x{i} + {height[i-1]} z{i} + {width[i-1]} - {width[i-1]} z{i} <= x{j} + {np.round(bound)} x{i}{j} + {np.round(bound)} y{i}{j};\n')
                        g.write(f'x{i} - {height[j-1]} z{j} - {width[j-1]} + {width[j-1]} z{j} >= x{j} - {np.round(bound)*1} + {np.round(bound)} x{i}{j} - {np.round(bound)} y{i}{j};\n')
                        g.write(f'y{i} + {width[i-1]} z{i} + {height[i-1]} - {height[i-1]} z{i} <= y{j} + {np.round(bound)*1} + {np.round(bound)} x{i}{j} - {np.round(bound)} y{i}{j};\n')
                        g.write(f'y{i} - {width[j-1]} z{j} - {height[j-1]} + {height[j-1]} z{j} >= y{j} - {np.round(bound)*2} + {np.round(bound)} x{i}{j} + {np.round(bound)} y{i}{j};\n\n\n')
            g.close()

    def hard_soft_nonoverlap(self):
        self.total_modules()
        if self.hard_exists and self.soft_exists:
            width_hard, height_hard = self.hard_module_width, self.hard_module_height
            gradient, intercept, bound = self.gradient, self.intercept, self.bound
            bound = self.upper_bound()
            g = open(self.output, 'a')
            g.write('/* Non-overlap constraints hard-soft */\n')
            for i in range(1, self.num_hard_modules+1):
                for j in range(self.num_hard_modules+1, self.num_hard_modules+1+self.num_soft_modules):
                    if j <= i:
                        continue
                    else:
                        g.write(f'x{i} + {height_hard[i-1]} z{i} + {width_hard[i-1]} - {width_hard[i-1]} z{i} <= x{j} + {np.round(bound)} x{i}{j} + {np.round(bound)} y{i}{j};\n')
                        g.write(f'x{i} - w{j} >= x{j} - {np.round(bound)*1} + {np.round(bound)} x{i}{j} - {np.round(bound)} y{i}{j};\n')
                        g.write(f'y{i} + {width_hard[i-1]} z{i} + {height_hard[i-1]} - {height_hard[i-1]} z{i} <= y{j} + {np.round(bound)*1} + {np.round(bound)} x{i}{j} - {np.round(bound)} y{i}{j};\n')
                        g.write(f'y{i} + {-1*gradient[j-self.num_hard_modules-1]} w{j} - {intercept[j-self.num_hard_modules-1]} >= y{j} - {np.round(bound)*2} + {np.round(bound)} x{i}{j} + {np.round(bound)} y{i}{j};\n\n\n')
            g.close()

    def soft_soft_nonoverlap(self):
        self.total_modules()
        if self.soft_exists:
            gradient, intercept, bound = self.gradient, self.intercept, self.bound
            g = open(self.output, 'a')
            g.write('/* Non-overlap constraints soft-soft */\n')
            for i in range(self.num_hard_modules+1, self.num_hard_modules+1+self.num_soft_modules):
                for j in range(self.num_hard_modules+1, self.num_hard_modules+1+self.num_soft_modules):
                    if j <= i:
                        continue
                    else:
                        g.write(f'x{i} + w{i} <= x{j} + {np.round(bound)} x{i}{j} + {np.round(bound)} y{i}{j};\n')
                        g.write(f'x{i} - w{j} >= x{j} - {np.round(bound)*1} + {np.round(bound)} x{i}{j} - {np.round(bound)} y{i}{j};\n')
                        g.write(f'y{i} - {-1*gradient[i-self.num_hard_modules-1]} w{i} + {intercept[i-self.num_hard_modules-1]} <= y{j} + {np.round(bound)} + {np.round(bound)} x{i}{j} - {np.round(bound)} y{i}{j};\n')
                        g.write(f'y{i} + {-1*gradient[j-self.num_hard_modules-1]} w{j} - {intercept[j-self.num_hard_modules-1]} >= y{j} - {np.round(bound)*2} + {np.round(bound)} x{i}{j} + {np.round(bound)} y{i}{j};\n')
            g.close()

    def variable_type_constraint(self):
        g = open(self.output, 'a')
        g.write('/* variable type constraints */\n')
        soft_module_widths = self.soft_module_dimension_range()[0]
        for i in range(1, self.num_hard_modules+self.num_soft_modules+1):
            g.write(f'x{i} >= 0;\n')
            g.write(f'y{i} >= 0;\n')
            g.write('x'+str(i)+' >= 0;\n')
        for i in range(self.num_hard_modules+1, self.num_hard_modules+self.num_soft_modules+1):
            g.write(f'w{i} >= {soft_module_widths[i-self.num_hard_modules-1, 0]};\n')
            g.write(f'w{i} <= {soft_module_widths[i-self.num_hard_modules-1, 1]};\n')
        g.write('\n\n')
        g.close()

    def chip_width_constraint(self):
        width_hard, height_hard = self.hard_module_width, self.hard_module_height
        g = open(self.output, 'a')
        g.write('/* chip width constraints */\n')
        for i in range(1, self.num_hard_modules+1):
            g.write(f'x{i} + {width_hard[i-1]} - {width_hard[i-1]} z{i} + {height_hard[i-1]} z{i} <= Y;\n')
        for i in range(self.num_hard_modules+1, self.num_hard_modules+1+self.num_soft_modules):
            g.write(f'x{i} + w{i} <= Y;\n')
        g.write('\n\n')
        g.close()

    def chip_height_constraint(self):
        width_hard, height_hard = self.hard_module_width, self.hard_module_height
        gradient, intercept = self.gradient, self.intercept
        g = open(self.output, 'a')
        g.write('/* chip height constraints */\n')
        for i in range(1, self.num_hard_modules+1):
            g.write(f'y{i} + {height_hard[i-1]} - {height_hard[i-1]} z{i} + {width_hard[i-1]} z{i} <= Y;\n')
        for i in range(self.num_hard_modules+1, self.num_hard_modules+1+self.num_soft_modules):
            g.write(f'y{i} - {-1*gradient[i-self.num_hard_modules-1]} w{i} + {intercept[i-self.num_hard_modules-1]} <= Y;\n')
            
        g.write('\n\n')
        g.close()

    def binary_constraints(self):
        g = open(self.output, 'a')
        g.write('/* variable type constraints */\n')
        g.write('bin ')
        for i in range(1, self.num_hard_modules+1+self.num_soft_modules):
            for j in range(1, self.num_hard_modules+1+self.num_soft_modules):
                if j <= i:
                    continue
                else:
                    if i == self.num_hard_modules+self.num_soft_modules-1 and j == self.num_hard_modules+self.num_soft_modules:
                        g.write(f'x{i}{j};\n')
                    else:
                        g.write(f'x{i}{j}, ')
        g.write('bin ')

        for i in range(1, self.num_hard_modules+1+self.num_soft_modules):
            for j in range(1, self.num_hard_modules+1+self.num_soft_modules):
                if j <= i:
                    continue
                else:
                    if i == self.num_hard_modules+self.num_soft_modules-1 and j == self.num_hard_modules+self.num_soft_modules:
                        g.write(f'y{i}{j};\n')
                    else:
                        g.write(f'y{i}{j}, ')
        g.write('bin ')
        for i in range(1, self.num_hard_modules+1):
            if i == self.num_hard_modules:
                g.write('z'+str(i)+';\n')
            else:
                g.write('z'+str(i)+', ')
        g.close()

    def create_ilp_file(self):
        self.objective()
        self.hard_hard_nonoverlap()
        self.hard_soft_nonoverlap()
        self.soft_soft_nonoverlap()
        self.variable_type_constraint()
        self.chip_width_constraint()
        self.chip_height_constraint()
        self.binary_constraints()