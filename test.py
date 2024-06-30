from utils.mip_utils import formulate_mip_from_file, MipSolvingConfig, solve_mip
import json
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='PP-Global', choices=['PP-Global', 'PP-Local'], help='Dataset to use')
    parser.add_argument('--data_idx', '-i', type=int, default=1, help='Index of the instance in the dataset')
    args = parser.parse_args()
    # check
    if args.data_idx < 1 or args.data_idx > 100:
        raise ValueError('Index must be in range 1-100')
    
    # load from file
    data_dir = os.path.join(args.dataset, f'{args.dataset.replace("-", "_")}_{args.data_idx}.mps')
    print(f'Loading data from {data_dir}')
    MIP_CONFIG = {
        'solver_type': 'scip',
        'obj_type': 'product_num',
        'time_limit': 3000,
        'show_log': True,
    }
    mip_model = formulate_mip_from_file(
        file_path=data_dir, 
        solving_config=MipSolvingConfig(**MIP_CONFIG)
    )
    # solve
    mip_results = solve_mip(
        solving_config=MipSolvingConfig(**MIP_CONFIG),
        model=mip_model
    )
    print(f'Solving time: {mip_results["solving_time"]:.2f}s, '
          f'Objective value: {mip_results["obj_value"]:.2f}')