def total_modules(lines):
    num_hard_modules, num_soft_modules = 0, 0
    for line in lines:
        if line[0:4] == 'hard':
            num_hard_modules = line.replace('hard - ', '')
            num_hard_modules = int(num_hard_modules.replace('\n', ''))
        elif line[0:4] == 'soft':
            num_soft_modules = line.replace('soft - ', '')
            num_soft_modules = int(num_soft_modules.replace('\n', ''))

    return num_hard_modules, num_soft_modules