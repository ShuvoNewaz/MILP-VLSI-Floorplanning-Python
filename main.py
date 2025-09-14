import argparse
import numpy as np
from src.solve.solve import *
from src.augment.augment import Augment
from src.augment.multi_threading import successiveAugmentation
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
import warnings

warnings.filterwarnings("ignore")


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_blocks', type=int, default=30, help='The number of blocks to be optimized.')
parser.add_argument('-u', '--underestimation', type=boolean_string, default=True)
parser.add_argument('-sa', '--successive_augmentation', type=boolean_string, default=False)
parser.add_argument('--runtime', type=int, default=10, help='The time the solver is given to solve a subproblem.')
parser.add_argument('-vis', '--visualize_superblock', type=boolean_string, default=True)
parser.add_argument('-lp', '--save_lp', type=boolean_string, default=True, help='Create an lp formatted file for use with the LPSolve tool.')
parser.add_argument('-size', '--sub_block_size', type=int, default=10, help='Size of the superblock')
args = parser.parse_args()

cwd = os.getcwd()
spec_files_dir = os.path.join(cwd, 'spec_files')
sa_files_dir = os.path.join(spec_files_dir, 'successive_augmentation', str(args.num_blocks))

file = f'{args.num_blocks}_block.ilp'

utilizations = []
if args.successive_augmentation:
    if os.path.exists(sa_files_dir):
        shutil.rmtree(sa_files_dir)
    os.makedirs(sa_files_dir, exist_ok=True)
    
    aug = Augment(file)
    aug.break_problem(sub_block_size=args.sub_block_size) # This breaks the large problem into several smaller subproblems
    num_augmentations = len(os.listdir(sa_files_dir))
    chip_heights, chip_widths = [], []

    # Initiaite multi-threading

    with ThreadPoolExecutor(max_workers=num_augmentations) as executor:
        futures = [executor.submit(successiveAugmentation, sa_files_dir, args.runtime,
                                   args.underestimation, args.num_blocks, i)
                                   for i in range(1, num_augmentations + 1)]
        for future in as_completed(futures):
            try:
                chip_height, chip_width, X, Y, Z, W, H, i, problem = future.result()
                chip_heights.append(chip_height)
                chip_widths.append(chip_width)
                problem.compute_utilization(chip_height, chip_width, H, W)
                problem.visualize(chip_height, chip_width, X, Y, Z, W, H, idx=i,
                                  sa=args.successive_augmentation,
                                  show_layout=args.visualize_superblock)
                utilizations.append(problem.utilization)
            except Exception as e:
                print(f"Error processing superblock: {e}")

    save_augmented_dimensions(args.num_blocks, chip_heights, chip_widths) # Creates a new source file from the optimized super-blocks

    # Solve for the entire problem using super-blocks
    src_file_path = os.path.join(sa_files_dir, f'{args.num_blocks}_blocks_sa.ilp')
else:
    src_file_path = os.path.join(spec_files_dir, file)
    utilizations.append(1)
problem = SolveILP(src_file_path,
                   underestimation=args.underestimation,
                   save_lp=args.save_lp)
problem.create_constraints()
chip_height, chip_width, X, Y, Z, W, H = problem.solve(run_time=args.runtime)
problem.compute_utilization(chip_height, chip_width, H, W, utilizations)
problem.visualize(chip_height, chip_width, X, Y, Z, W, H, glob=True,
                  sa=args.successive_augmentation)
save_final_dimensions(chip_height, chip_width, args.num_blocks,
                      args.successive_augmentation)