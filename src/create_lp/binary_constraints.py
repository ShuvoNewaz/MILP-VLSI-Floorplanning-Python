def binary_constraints(output,
                       num_hard_modules,
                       num_total_modules):
    g = open(output, 'a')
    g.write('/* variable type constraints */\n')
    g.write('bin ')
    for i in range(1, num_total_modules + 1):
        for j in range(1, num_total_modules + 1):
            if j <= i:
                continue
            else:
                if i == num_total_modules - 1 and j == num_total_modules:
                    g.write(f'x{i}{j};\n')
                else:
                    g.write(f'x{i}{j}, ')
    g.write('bin ')

    for i in range(1, num_total_modules + 1):
        for j in range(1, num_total_modules + 1):
            if j <= i:
                continue
            else:
                if i == num_total_modules - 1 and j == num_total_modules:
                    g.write(f'y{i}{j};\n')
                else:
                    g.write(f'y{i}{j}, ')
    g.write('bin ')
    for i in range(1, num_hard_modules+1):
        if i == num_hard_modules:
            g.write('z'+str(i)+';\n')
        else:
            g.write('z'+str(i)+', ')
    g.close()