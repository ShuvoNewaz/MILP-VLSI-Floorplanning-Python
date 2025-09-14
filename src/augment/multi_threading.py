import os
from src.solve.solve import SolveILP


def successiveAugmentation(sa_files_dir, runtime,
                           underestimation, num_blocks, i):
    src_file_path = os.path.join(sa_files_dir,
                                 f'{num_blocks}_{i}.ilp') # Takes a super-block
    problem = SolveILP(src_file_path,
                       underestimation=underestimation,
                       save_lp=False) # Solves for the super-block
    problem.create_constraints()
    chip_height, chip_width, X, Y, Z, W, H = problem.solve(run_time=runtime)

    return chip_height, chip_width, X, Y, Z, W, H, i, problem