import numpy as np


def soft_module_properties(lines,
                           num_soft_modules):
    if num_soft_modules > 0:
        area, min_aspect, max_aspect = [], [], []

        for line_count in range(len(lines)):
            if lines[line_count][0:4] == 'soft':
                break
        for i in range(line_count+1, line_count+1+num_soft_modules):
            first_comma_ind = lines[i].index(',')
            module_area = (lines[i][:first_comma_ind])
            line_after_comma = lines[i][first_comma_ind+1:]
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