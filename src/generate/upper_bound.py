import numpy as np


def upper_bound(hard_module_width,
                hard_module_height,
                soft_module_width_range,
                soft_module_height_range):
    W_hard = np.maximum(hard_module_width, hard_module_height).sum()
    H_hard = W_hard
    W_soft = soft_module_width_range[:, 1].sum()
    H_soft = soft_module_height_range[:, 1].sum()

    W = W_hard + W_soft
    H = H_hard + H_soft

    return np.max([W, H])