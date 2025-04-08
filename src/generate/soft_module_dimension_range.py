import numpy as np


def soft_module_dimension_range(soft_exists,
                                underestimation,
                                area,
                                min_aspect,
                                max_aspect):
    if soft_exists:
        min_w = np.sqrt(area * min_aspect)[:, np.newaxis]
        max_w = np.sqrt(area * max_aspect)[:, np.newaxis]
        area = area[:, np.newaxis]

        soft_module_width_range = np.concatenate((min_w, max_w), axis=1)
        if underestimation:
            soft_module_height_range = area / max_w + (max_w - soft_module_width_range) * area / max_w ** 2
        else:
            soft_module_height_range = area / soft_module_width_range

        return soft_module_width_range, soft_module_height_range[:, ::-1]
    else:
        soft_module_width_range = np.array([[0, 0]])
        soft_module_height_range = np.array([[0, 0]])

        return soft_module_width_range, soft_module_height_range