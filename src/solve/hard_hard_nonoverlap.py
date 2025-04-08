def hard_hard_nonoverlap(hard_exists,
                         num_hard_modules,
                         hard_module_width,
                         hard_module_height,
                         bound,
                         x, y, x_ij, y_ij, z,
                         constraints):
    if hard_exists:
        for i in range(num_hard_modules):
            for j in range(num_hard_modules):
                if j > i:
                    constraints.append(x[i] + 
                                       z[i] * hard_module_height[i] + 
                                       (1-z[i]) * hard_module_width[i] <= 
                                       x[j] + bound * (x_ij[i, j] + y_ij[i, j]))
                    constraints.append(x[i] - 
                                       z[j] * hard_module_height[j] - 
                                       (1-z[j]) * hard_module_width[j] >= 
                                       x[j] - bound * (1 - x_ij[i, j] + y_ij[i, j]))
                    constraints.append(y[i] + 
                                       z[i] * hard_module_width[i] + 
                                       (1-z[i]) * hard_module_height[i] <= 
                                       y[j] + bound * (1 + x_ij[i, j] - y_ij[i, j]))
                    constraints.append(y[i] - 
                                       z[j] * hard_module_width[j] - 
                                       (1-z[j]) * hard_module_height[j] >= 
                                       y[j] - bound * (2 - x_ij[i, j] - y_ij[i, j]))
