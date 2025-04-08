def chip_height_constraint(output,
                        hard_module_width,
                        hard_module_height,
                        num_hard_modules,
                        num_total_modules,
                        gradient,
                        intercept):
    width_hard, height_hard = hard_module_width, hard_module_height
    g = open(output, 'a')
    g.write('/* chip height constraints */\n')
    for i in range(1, num_hard_modules+1):
        g.write(f'y{i} + {height_hard[i-1]} - {height_hard[i-1]} z{i} + {width_hard[i-1]} z{i} <= Y;\n')
    for i in range(num_hard_modules + 1, num_total_modules + 1):
        g.write(f'y{i} - {-1*gradient[i-num_hard_modules-1]} w{i} + {intercept[i-num_hard_modules-1]} <= Y;\n')
        
    g.write('\n\n')
    g.close()