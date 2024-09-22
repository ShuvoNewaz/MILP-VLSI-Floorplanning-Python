import numpy as np
import os


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
        self.num_hard_modules, self.num_soft_modules = self.total_modules()


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

    def break_problem(self, sub_block_size=10):
        num_hard_modules, num_soft_modules = self.total_modules()
        num_total_modules = num_hard_modules + num_soft_modules
        hard_module_width, hard_module_height = self.hard_module_dimension()
        area, min_aspect, max_aspect = self.soft_module_properties()

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