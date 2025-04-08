import numpy as np


def soft_soft_nonoverlap(soft_exists,
                        output,
                        num_hard_modules,
                        num_total_modules,
                        gradient,
                        intercept,
                        bound):
    if soft_exists:
        gradient, intercept, bound = gradient, intercept, bound
        g = open(output, 'a')
        g.write('/* Non-overlap constraints soft-soft */\n')
        for i in range(num_hard_modules + 1, num_total_modules + 1):
            for j in range(num_hard_modules + 1, num_total_modules + 1):
                if j <= i:
                    continue
                else:
                    g.write(f'x{i} + w{i} <= x{j} + {np.round(bound)} x{i}{j} + {np.round(bound)} y{i}{j};\n')
                    g.write(f'x{i} - w{j} >= x{j} - {np.round(bound)*1} + {np.round(bound)} x{i}{j} - {np.round(bound)} y{i}{j};\n')
                    g.write(f'y{i} - {-1*gradient[i-num_hard_modules-1]} w{i} + {intercept[i-num_hard_modules-1]} <= y{j} + {np.round(bound)} + {np.round(bound)} x{i}{j} - {np.round(bound)} y{i}{j};\n')
                    g.write(f'y{i} + {-1*gradient[j-num_hard_modules-1]} w{j} - {intercept[j-num_hard_modules-1]} >= y{j} - {np.round(bound)*2} + {np.round(bound)} x{i}{j} + {np.round(bound)} y{i}{j};\n')
        g.close()