import numpy as np


def hard_soft_nonoverlap(hard_exists,
                        soft_exists,
                        output,
                        num_hard_modules,
                        num_total_modules,
                        hard_module_width,
                        hard_module_height,
                        gradient,
                        intercept,
                        bound):
    if hard_exists and soft_exists:
        g = open(output, 'a')
        g.write('/* Non-overlap constraints hard-soft */\n')
        for i in range(1, num_hard_modules + 1):
            for j in range(num_hard_modules + 1,
                           num_total_modules + 1):
                if j <= i:
                    continue
                else:
                    g.write(f'x{i} + {hard_module_height[i-1]} z{i} + {hard_module_width[i-1]} - {hard_module_width[i-1]} z{i} <= x{j} + {np.round(bound)} x{i}{j} + {np.round(bound)} y{i}{j};\n')
                    g.write(f'x{i} - w{j} >= x{j} - {np.round(bound)*1} + {np.round(bound)} x{i}{j} - {np.round(bound)} y{i}{j};\n')
                    g.write(f'y{i} + {hard_module_width[i-1]} z{i} + {hard_module_height[i-1]} - {hard_module_height[i-1]} z{i} <= y{j} + {np.round(bound)*1} + {np.round(bound)} x{i}{j} - {np.round(bound)} y{i}{j};\n')
                    g.write(f'y{i} + {-1*gradient[j-num_hard_modules-1]} w{j} - {intercept[j-num_hard_modules-1]} >= y{j} - {np.round(bound)*2} + {np.round(bound)} x{i}{j} + {np.round(bound)} y{i}{j};\n\n\n')
        g.close()