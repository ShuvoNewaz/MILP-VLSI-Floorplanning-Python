def soft_soft_nonoverlap(soft_exists,
                         num_hard_modules,
                         num_total_modules,
                         gradient, intercept,
                         bound,
                         x, y, x_ij, y_ij, w,
                         constraints):
    if soft_exists:
        for i in range(num_hard_modules, num_total_modules):
            for j in range(num_hard_modules, num_total_modules):
                if j > i:
                    constraints.append(x[i] + 
                                       w[i-num_hard_modules] <= 
                                       x[j] + bound * (x_ij[i, j] + y_ij[i, j]))
                    constraints.append(x[i] - 
                                       w[j-num_hard_modules] >= 
                                       x[j] - bound * (1 - x_ij[i, j] + y_ij[i, j]))
                    constraints.append(y[i] + 
                                       (gradient[i-num_hard_modules] * w[i-num_hard_modules] + 
                                        intercept[i-num_hard_modules]) <= 
                                        y[j] + bound * (1 + x_ij[i, j] - y_ij[i, j]))
                    constraints.append(y[i] - 
                                       (gradient[j-num_hard_modules] * w[j-num_hard_modules] + 
                                        intercept[j-num_hard_modules]) >= 
                                        y[j] - bound * (2 - x_ij[i, j] - y_ij[i, j]))
