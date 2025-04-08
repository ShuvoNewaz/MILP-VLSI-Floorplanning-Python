def variable_type_constraint(output,
                            soft_module_width_range,
                            num_hard_modules,
                            num_total_modules):
    g = open(output, 'a')
    g.write('/* variable type constraints */\n')
    for i in range(1, num_total_modules + 1):
        g.write(f'x{i} >= 0;\n')
        g.write(f'y{i} >= 0;\n')
    for i in range(num_hard_modules + 1, num_total_modules + 1):
        g.write(f'w{i} >= {soft_module_width_range[i-num_hard_modules-1, 0]};\n')
        g.write(f'w{i} <= {soft_module_width_range[i-num_hard_modules-1, 1]};\n')
    g.write('\n\n')
    g.close()