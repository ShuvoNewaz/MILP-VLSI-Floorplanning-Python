def hard_soft_nonoverlap(hard_exists,
                         soft_exists,
                         num_hard_modules,
                         num_total_modules,
                         hard_module_width,
                         hard_module_height,
                         gradient, intercept,
                         bound,
                         x, y, x_ij, y_ij, z, w,
                         constraints):
    if hard_exists and soft_exists:
        for i in range(num_hard_modules):
            for j in range(num_hard_modules, num_total_modules):
                if j > i:
                    constraints.append(x[i] + 
                                       z[i] * hard_module_height[i] + 
                                       (1-z[i]) * hard_module_width[i] <= 
                                       x[j] + bound * (x_ij[i, j] + y_ij[i, j]))
                    constraints.append(x[i] - 
                                       w[j - num_hard_modules] >= 
                                       x[j] - bound * (1 - x_ij[i, j] + y_ij[i, j]))
                    constraints.append(y[i] + 
                                       z[i] * hard_module_width[i] + 
                                       (1-z[i]) * hard_module_height[i] <= 
                                       y[j] + bound * (1 + x_ij[i, j] - y_ij[i, j]))
                    constraints.append(y[i] - 
                                       (gradient[j-num_hard_modules] * w[j-num_hard_modules] + 
                                        intercept[j-num_hard_modules]) >= 
                                        y[j] - bound * (2 - x_ij[i, j] - y_ij[i, j]))
