import argparse

parser = argparse.ArgumentParser(description='Process some fixed parameters.')

parser.add_argument(
    '--data_dir', default='/home/cxl173430/data/DATASETS', 
    type=str, help='dataset directory')

parser.add_argument(
    "--output_folder", default="/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results/", 
    type=str, help="where results will be saved.")

parser.add_argument(
    "--saved_spec_dir", default="Debug", 
    type=str, help="specific experiment path.")

parser.add_argument(
    '--seed', default=42, 
    type=int, help='set random seed')

parser.add_argument(
    '--gpu', default=0, 
    type=int, help='The index of the gpu used')

# parser.add_argument(
    # '--backbone', default='EfficientNet', 
    # type=str, help='backbone models')

# parser.add_argument(
#     '--batch_size', default=64, 
#     type=int, help='The index of the gpu used')

##### Hyperparameters #####
parser.add_argument(
    '--epochs', default=100, 
    type=int, help='The number of composite elements'
    )

parser.add_argument(
    '--num_comp', default=5, 
    type=int, help='The number of composite elements'
    )

parser.add_argument(
    '--gauss_kernel_size', default=11, 
    type=int, help='gauss_kernel_size'
    )

parser.add_argument(
    '--init_lr', default=0.1, 
    type=float, help='set init learning rate'
    )

parser.add_argument(
    '--kl_lam_GDD', default=1.0,
    type=float, help="weight for regularizer: l2 norm"
)

parser.add_argument(
    '--entropy_lam', default=0.0,
    type=float, help="weight for regularizer: entropy"
)

parser.add_argument(
    '--entropy_lam_Dir', default=0.0,
    type=float, help="weight for regularizer: entropy of Dirichlet distribution for GDD"
)

parser.add_argument(
    '--entropy_lam_GDD', default=0.0,
    type=float, help="weight for regularizer: entropy of Group Dirichlet distribution for GDD"
)

# parser.add_argument(
#     '--epochs_stage_1', default=25, 
#     type=int, help='the first stage of epochs'
#     )
# parser.add_argument(
#     '--epochs_stage_2', default=10, 
#     type=int, help='the second stage of epochs'
#     )


