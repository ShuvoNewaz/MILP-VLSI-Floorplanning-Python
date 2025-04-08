import numpy as np


def linear_approximation(soft_exists,
                        underestimation,
                        area,
                        soft_module_width_range,
                        soft_module_height_range):
    if soft_exists:
        area = area
        min_w = soft_module_width_range[:, 0]
        max_w = soft_module_width_range[:, 1]
        min_h = soft_module_height_range[:, 0]
        max_h = soft_module_height_range[:, 1]

        if underestimation:
            gradient = - area / max_w ** 2
            intercept = 2 * area / max_w
        else:
            gradient = (max_h - min_h) / (min_w - max_w)
            intercept = max_h - gradient * min_w

        return gradient, intercept
    else:

        return 0, 0