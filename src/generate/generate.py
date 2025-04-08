import numpy as np
import os
from typing import List
from src.generate.total_modules import total_modules
from src.generate.hard_module_dimension import hard_module_dimension
from src.generate.soft_module_properties import soft_module_properties
from src.generate.soft_module_dimension_range import soft_module_dimension_range
from src.generate.linear_approximation import linear_approximation
from src.generate.upper_bound import upper_bound
from src.create_lp.create_lp_file import create_lp_file


cwd = os.getcwd()
spec_files_dir = os.path.join(cwd, 'spec_files') # Contains the initial specifications
lp_solve_files_dir = os.path.join(cwd, 'lp_solve_files') # Contains the source file for LPSolve tool


class GenerateProblem:

    def __init__(self, file, underestimation=True, save_lp=True):
        """
        args:
            file: The provided *.ilp file (str)
            num_blocks: Number of blocks to be optimized (int)
            underestimation: Whether or not we are considering underestimation (bool)
        """
        self.underestimation = underestimation
        lines = []
        with open(file) as f:
            for line in f:
                lines.append(line)
        self.lines = lines
        
        self.num_hard_modules, self.num_soft_modules = total_modules(lines)
        self.hard_exists = self.num_hard_modules > 0
        self.soft_exists = self.num_soft_modules > 0
        self.num_total_modules = self.num_hard_modules + self.num_soft_modules
        self.hard_module_width, self.hard_module_height = hard_module_dimension(lines,
                                                                self.num_hard_modules)
        self.area, self.min_aspect, self.max_aspect = soft_module_properties(lines,
                                                                self.num_soft_modules)
        self.soft_module_width_range, self.soft_module_height_range = \
            soft_module_dimension_range(self.soft_exists,
                                        self.underestimation,
                                        self.area,
                                        self.min_aspect,
                                        self.max_aspect)
        self.gradient, self.intercept = \
            linear_approximation(self.soft_exists,
                                self.underestimation,
                                self.area,
                                self.soft_module_width_range,
                                self.soft_module_height_range)
        self.bound = upper_bound(self.hard_module_width,
                                    self.hard_module_height,
                                    self.soft_module_width_range,
                                    self.soft_module_height_range)
        output = os.path.join(lp_solve_files_dir, 
                              f'{self.num_total_modules}_blocks_constraints.lp')
        if save_lp:
            print(f"Saving LP file to {output}")
            create_lp_file(lp_solve_files_dir,
                           output,
                           self.hard_exists,
                           self.soft_exists,
                           self.num_hard_modules,
                           self.num_total_modules,
                           self.hard_module_width,
                           self.hard_module_height,
                           self.bound,
                           self.gradient,
                           self.intercept,
                           self.soft_module_width_range)