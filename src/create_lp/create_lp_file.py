import os
from src.create_lp.objective import objective
from src.create_lp.hard_hard_nonoverlap import hard_hard_nonoverlap
from src.create_lp.hard_soft_nonoverlap import hard_soft_nonoverlap
from src.create_lp.soft_soft_nonoverlap import soft_soft_nonoverlap
from src.create_lp.variable_type_constraint import variable_type_constraint
from src.create_lp.chip_height_constraint import chip_height_constraint
from src.create_lp.chip_width_constraint import chip_width_constraint
from src.create_lp.binary_constraints import binary_constraints


def create_lp_file(lp_solve_files_dir,
                    output,
                    hard_exists,
                    soft_exists,
                    num_hard_modules,
                    num_total_modules,
                    hard_module_width,
                    hard_module_height,
                    bound,
                    gradient,
                    intercept,
                    soft_module_width_range):
    os.makedirs(lp_solve_files_dir, exist_ok=True)
    objective(output)
    hard_hard_nonoverlap(hard_exists,
                         output,
                         num_hard_modules,
                         hard_module_width,
                         hard_module_height,
                         bound)
    hard_soft_nonoverlap(hard_exists,
                        soft_exists,
                        output,
                        num_hard_modules,
                        num_total_modules,
                        hard_module_width,
                        hard_module_height,
                        gradient,
                        intercept,
                        bound)
    soft_soft_nonoverlap(soft_exists,
                        output,
                        num_hard_modules,
                        num_total_modules,
                        gradient,
                        intercept,
                        bound)
    variable_type_constraint(output,
                            soft_module_width_range,
                            num_hard_modules,
                            num_total_modules)
    chip_width_constraint(output,
                        hard_module_width,
                        hard_module_height,
                        num_hard_modules,
                        num_total_modules)
    chip_height_constraint(output,
                        hard_module_width,
                        hard_module_height,
                        num_hard_modules,
                        num_total_modules,
                        gradient,
                        intercept)
    binary_constraints(output,
                       num_hard_modules,
                       num_total_modules)