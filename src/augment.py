import numpy as np
import os
from src.generate.total_modules import total_modules
from src.generate.hard_module_dimension import hard_module_dimension
from src.generate.soft_module_properties import soft_module_properties


cwd = os.getcwd()
spec_files_dir = os.path.join(cwd, 'spec_files') # Contains the initial specifications
sa_files_dir = os.path.join(spec_files_dir, 'successive_augmentation')


class Augment:
    def __init__(self, file, underestimation=True):
        self.hard_exists = False
        self.soft_exists = False
        self.underestimation = underestimation

        num_blocks = file.split('_')[0]
        spec_file = os.path.join(spec_files_dir, file)
        lines = []
        with open(spec_file) as f:
            for line in f:
                lines.append(line)
        self.lines = lines
        self.sa_files_dir = os.path.join(sa_files_dir, num_blocks)
        os.makedirs(self.sa_files_dir, exist_ok=True)
        self.sa_file_prefix = os.path.join(self.sa_files_dir, f'{num_blocks}')
        self.num_hard_modules, self.num_soft_modules = total_modules(lines)
        self.num_total_modules = self.num_hard_modules + self.num_soft_modules
        self.hard_module_width, self.hard_module_height = hard_module_dimension(lines,
                                                                self.num_hard_modules)
        self.area, self.min_aspect, self.max_aspect = soft_module_properties(lines,
                                                                self.num_soft_modules)

    def break_problem(self, sub_block_size=10):
        num_hard_modules, num_soft_modules, num_total_modules = \
            self.num_hard_modules, self.num_soft_modules, self.num_total_modules
        hard_module_width, hard_module_height = self.hard_module_width, self.hard_module_height
        area, min_aspect, max_aspect = self.area, self.min_aspect, self.max_aspect

        if num_total_modules > sub_block_size:
            num_subblocks = int(np.ceil(num_total_modules / sub_block_size))
            soft_count = 0
            for i in range(num_subblocks):
                modules_in_subblock = min(sub_block_size, num_total_modules - i * sub_block_size)
                hard_left = num_hard_modules - sub_block_size*i
                if hard_left >= modules_in_subblock: # Submodule has all hard modules
                    g = open(f'{self.sa_file_prefix}_{i+1}.ilp', 'w')
                    g.write(f'hard - {modules_in_subblock}\n')
                    g.close()
                    g = open(f'{self.sa_file_prefix}_{i+1}.ilp', 'a')
                    for j in range(modules_in_subblock):
                        g.write(f'{hard_module_width[i*modules_in_subblock+j]},{hard_module_height[i*modules_in_subblock+j]}\n')
                    g.close()
                elif modules_in_subblock > hard_left > 0: # Submodule has a mixture of hard and soft modules
                    g = open(f'{self.sa_file_prefix}_{i+1}.ilp', 'w')
                    g.write(f'hard - {hard_left}\n')
                    g.close()
                    g = open(f'{self.sa_file_prefix}_{i+1}.ilp', 'a')
                    for j in range(hard_left):
                        g.write(f'{hard_module_width[i*modules_in_subblock+j]},{hard_module_height[i*modules_in_subblock+j]}\n')
                    # if hard_left + num_soft_modules >= modules_in_subblock:
                    if num_soft_modules > 0:
                        g.write(f'\nsoft - {modules_in_subblock - hard_left}\n')
                        for j in range(modules_in_subblock - hard_left):
                            g.write(f'{area[j]},{min_aspect[j]},{max_aspect[j]}\n')
                        g.close()
                        area = area[j:]
                        min_aspect = min_aspect[j:]
                        max_aspect = max_aspect[j:]
                    soft_left = num_soft_modules - (modules_in_subblock - hard_left)
                elif hard_left <= 0:
                    soft_left = num_soft_modules - modules_in_subblock * soft_count
                    g = open(f'{self.sa_file_prefix}_{i+1}.ilp', 'w')
                    g.write(f'soft - {modules_in_subblock}\n')
                    g.close()
                    g = open(f'{self.sa_file_prefix}_{i+1}.ilp', 'a')
                    if soft_left > 0:
                        for j in range(modules_in_subblock):
                            g.write(f'{area[soft_count*modules_in_subblock+j]},{min_aspect[soft_count*modules_in_subblock+j]},{max_aspect[soft_count*modules_in_subblock+j]}\n')
                        g.close()
                        soft_count += 1