import numpy as np


def hard_module_dimension(lines,
                          num_hard_modules):
    hard_exists = num_hard_modules > 0
    if hard_exists:
        hard_module_width = []
        hard_module_height = []
        for line_count in range(len(lines)):
            if lines[line_count][0:4] == 'hard':
                break
        for i in range (line_count+1, line_count+1+num_hard_modules):
            comma_ind = lines[i].index(',')
            width = lines[i][:comma_ind]
            height = lines[i][comma_ind+1:]
            hard_module_width.append(float(width))
            hard_module_height.append(float(height))

        hard_module_width = np.array(hard_module_width)
        hard_module_height = np.array(hard_module_height)
    else:
        hard_module_width, hard_module_height = 0, 0

    return hard_module_width, hard_module_height