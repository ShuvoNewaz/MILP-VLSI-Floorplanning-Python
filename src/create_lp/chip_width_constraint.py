def chip_width_constraint(output,
                        hard_module_width,
                        hard_module_height,
                        num_hard_modules,
                        num_total_modules):
    width_hard, height_hard = hard_module_width, hard_module_height
    g = open(output, 'a')
    g.write('/* chip width constraints */\n')
    for i in range(1, num_hard_modules+1):
        g.write(f'x{i} + {width_hard[i-1]} - {width_hard[i-1]} z{i} + {height_hard[i-1]} z{i} <= Y;\n')
    for i in range(num_hard_modules + 1, num_total_modules + 1):
        g.write(f'x{i} + w{i} <= Y;\n')
    g.write('\n\n')
    g.close()