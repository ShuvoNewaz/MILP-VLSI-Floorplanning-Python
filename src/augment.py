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
        num_hard_modules = self.total_modules()[0]
        if num_hard_modules > 0:
            hard_module_width = []
            hard_module_height = []
            for line_count in range(len(self.lines)):
                if self.lines[line_count][0:4] == 'hard':
                    break
            for i in range (line_count+1, line_count+1+num_hard_modules):
                comma_ind = self.lines[i].index(',')
                width = self.lines[i][:comma_ind]
                height = self.lines[i][comma_ind+1:]
                hard_module_width.append(float(width))
                hard_module_height.append(float(height))

            hard_module_width = np.array(hard_module_width)[:, np.newaxis]
            hard_module_height = np.array(hard_module_height)[:, np.newaxis]
        else:
            return np.array([0, 0])

        return np.concatenate((hard_module_width, hard_module_height), axis=1)

    def soft_module_properties(self):
        num_soft_modules = self.total_modules()[1]
        if num_soft_modules > 0:
            area, min_aspect, max_aspect = [], [], []

            for line_count in range(len(self.lines)):
                if self.lines[line_count][0:4] == 'soft':
                    break
            for i in range(line_count+1, line_count+1+num_soft_modules):
                first_comma_ind = self.lines[i].index(',')
                module_area = (self.lines[i][:first_comma_ind])
                line_after_comma = self.lines[i][first_comma_ind+1:]
                second_comma_ind = line_after_comma.index(',')
                area.append(float(module_area))
                module_min_aspect = line_after_comma[:second_comma_ind]
                module_max_aspect = line_after_comma[second_comma_ind+1:]
                min_aspect.append(float(module_min_aspect))
                max_aspect.append(float(module_max_aspect))

            area = np.array(area)[:, np.newaxis]
            min_aspect = np.array(min_aspect)[:, np.newaxis]
            max_aspect = np.array(max_aspect)[:, np.newaxis]
        else:
            return np.array([0, 0, 0])

        return np.concatenate((area, min_aspect, max_aspect), axis=1)

    def break_problem(self):
        num_hard_modules, num_soft_modules = self.total_modules()
        num_total_modules = num_hard_modules + num_soft_modules
        hard_module_dimension = self.hard_module_dimension()
        soft_module_properties = self.soft_module_properties()

        if num_total_modules > 10:
            if num_total_modules % 10 == 0:
                modules_in_subblock = 10
                num_subblocks = int(num_total_modules / 10)
                soft_count = 0
                for i in range(num_subblocks):
                    hard_left = num_hard_modules - modules_in_subblock*i
                    if hard_left >= modules_in_subblock:
                        g = open(f'{self.sa_file_prefix}_{i+1}.ilp', 'w')
                        g.write(f'hard - {modules_in_subblock}\n')
                        g.close()
                        g = open(f'{self.sa_file_prefix}_{i+1}.ilp', 'a')
                        for j in range(modules_in_subblock):
                            g.write(f'{hard_module_dimension[i*modules_in_subblock+j][0]},{hard_module_dimension[i*modules_in_subblock+j][1]}\n')
                        g.close()
                    elif modules_in_subblock > hard_left > 0:
                        g = open(f'{self.sa_file_prefix}_{i+1}.ilp', 'w')
                        g.write(f'hard - {hard_left}\n')
                        g.close()
                        g = open(f'{self.sa_file_prefix}_{i+1}.ilp', 'a')
                        for j in range(int(hard_left)):
                            g.write(f'{hard_module_dimension[i*modules_in_subblock+j][0]},{hard_module_dimension[i*modules_in_subblock+j][1]}\n')
                        if hard_left + num_soft_modules >= modules_in_subblock:
                            g.write(f'\nsoft - {modules_in_subblock - hard_left}\n')
                            for j in range(modules_in_subblock - hard_left):
                                g.write(f'{soft_module_properties[j][0]},{soft_module_properties[j][1]},{soft_module_properties[j][2]}\n')
                            g.close()
                            soft_module_properties = np.delete(soft_module_properties, slice(0, modules_in_subblock - hard_left), axis=0)
                        soft_left = num_soft_modules - (modules_in_subblock - hard_left)
                    elif hard_left <= 0:
                        soft_left = num_soft_modules - modules_in_subblock * soft_count
                        g = open(f'{self.sa_file_prefix}_{i+1}.ilp', 'w')
                        g.write(f'soft - {modules_in_subblock}\n')
                        g.close()
                        g = open(f'{self.sa_file_prefix}_{i+1}.ilp', 'a')
                        if soft_left > 0:
                            for j in range(modules_in_subblock):
                                g.write(str(soft_module_properties[soft_count*modules_in_subblock+j][0])+','+str(soft_module_properties[soft_count*modules_in_subblock+j][1])+','+str(soft_module_properties[soft_count*modules_in_subblock+j][2])+'\n')
                            g.close()
                            soft_count += 1
