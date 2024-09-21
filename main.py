import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.solve import SolveILP
from src.augment import Augment
import os
import shutil


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
parser.add_argument('-lp', '--lp_solve', type=boolean_string, default=True, help='Create an lp formatted file for use with the LPSolve tool.')
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
    if args.num_blocks < 10:
        raise ValueError('Successive augmentation does not support system with fewer than 10 blocks.')
    aug = Augment(file)
    aug.break_problem() # This breaks the large problem into several smaller subproblems
    num_augmentations = len(os.listdir(sa_files_dir))
    bounds = []
    for i in range(1, num_augmentations+1):
        src_file_path = os.path.join(sa_files_dir, f'{args.num_blocks}_{i}.ilp') # Takes a super-block
        problem = SolveILP(src_file_path, args.num_blocks, underestimation=args.underestimation) # Solves for the super-block
        problem.create_constraints()
        bound, X, Y, Z, W, H = problem.solve(run_time=args.runtime)
        bounds.append(bound)
        problem.visualize(bound, X, Y, Z, W, H, idx=i, sa=args.successive_augmentation, show_layout=args.visualize_superblock)
        utilizations.append(problem.utilization)
    problem.save_augmented_dimensions(bounds) # Creates a new source file from the optimized super-blocks

    # Solve for the entire problem using super-blocks
    src_file_path = os.path.join(sa_files_dir, f'{args.num_blocks}_blocks_sa.ilp')
else:
    src_file_path = os.path.join(spec_files_dir, file)
problem = SolveILP(src_file_path, args.num_blocks, underestimation=args.underestimation)
problem.create_constraints()
bound, X, Y, Z, W, H = problem.solve(run_time=args.runtime)
problem.visualize(bound, X, Y, Z, W, H, glob=True, sa=args.successive_augmentation, utilizations=utilizations)
problem.save_final_dimensions(bound, args.num_blocks, args.successive_augmentation)

if args.lp_solve:
    problem.problem.create_ilp_file()