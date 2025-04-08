def other_constraints(hard_exists,
                      soft_exists,
                      num_hard_modules,
                      num_soft_modules,
                      num_total_modules,
                      hard_module_width,
                      hard_module_height,
                      gradient, intercept,
                      soft_module_width_range,
                      x, y, x_ij, y_ij, z, w, Y,
                      constraints):
    for xVal, yVal in zip(x, y):
        constraints.append(xVal >= 0)
        constraints.append(yVal >= 0)

    for i in range(num_soft_modules):
        w_min = soft_module_width_range[i, 0]
        w_max = soft_module_width_range[i, 1]

        constraints.append(w_min <= w[i])
        constraints.append(w[i] <= w_max)

    for i in range(num_total_modules):
        for j in range(num_total_modules):
            if j > i:
                constraints.append(0 <= x_ij[i, j])
                constraints.append(x_ij[i, j] <= 1)
                constraints.append(0 <= y_ij[i, j])
                constraints.append(y_ij[i, j] <= 1)

    if hard_exists:
        for Z in z:
            constraints.append(0 <= Z)
            constraints.append(Z <= 1)

        for i in range(num_hard_modules):
            constraints.append(x[i] + 
                               z[i] * hard_module_height[i] + 
                               (1-z[i]) * hard_module_width[i] <= 
                               Y)
            constraints.append(y[i] + 
                               z[i] * hard_module_width[i] + 
                               (1-z[i]) * hard_module_height[i] <= 
                               Y)

    if soft_exists:
        for i in range(num_hard_modules, num_total_modules):
            constraints.append(x[i] + 
                               w[i-num_hard_modules] <= 
                               Y)
            constraints.append(y[i] + 
                               gradient[i-num_hard_modules] * w[i-num_hard_modules] + 
                               intercept[i-num_hard_modules] <= 
                               Y)