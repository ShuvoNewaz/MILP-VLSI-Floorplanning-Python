import numpy as np


def hard_hard_nonoverlap(hard_exists,
                         output,
                         num_hard_modules,
                         hard_module_width,
                         hard_module_height,
                         bound):
    if hard_exists:
        width, height = hard_module_width, hard_module_height
        g = open(output, 'a')
        g.write('/* Non-overlap constraints hard-hard */\n')
        for i in range(1, num_hard_modules+1):
            for j in range(1, num_hard_modules+1):
                if j <= i:
                    continue
                else:
                    g.write(f'x{i} + {height[i-1]} z{i} + {width[i-1]} - {width[i-1]} z{i} <= x{j} + {np.round(bound)} x{i}{j} + {np.round(bound)} y{i}{j};\n')
                    g.write(f'x{i} - {height[j-1]} z{j} - {width[j-1]} + {width[j-1]} z{j} >= x{j} - {np.round(bound)*1} + {np.round(bound)} x{i}{j} - {np.round(bound)} y{i}{j};\n')
                    g.write(f'y{i} + {width[i-1]} z{i} + {height[i-1]} - {height[i-1]} z{i} <= y{j} + {np.round(bound)*1} + {np.round(bound)} x{i}{j} - {np.round(bound)} y{i}{j};\n')
                    g.write(f'y{i} - {width[j-1]} z{j} - {height[j-1]} + {height[j-1]} z{j} >= y{j} - {np.round(bound)*2} + {np.round(bound)} x{i}{j} + {np.round(bound)} y{i}{j};\n\n\n')
        g.close()