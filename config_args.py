import argparse

parser = argparse.ArgumentParser(description='Process some fixed parameters.')

parser.add_argument(
    '--data_dir', default='/data/cxl173430/HyperEvidentialNN/TinyImageNet/', 
    type=str, help='dataset directory'
    )
# PDML2: '/home/cxl173430/Documents/projects/uncertainty_Related/HENN_originalNotebooks/'
# PDML4: '/home/cxl173430/Projects/uncertainty_Related/HENN_originalNotebooks/'

parser.add_argument(
    "--output_folder", default="/data/cxl173430/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results", 
    type=str, help="where results will be saved."
)
# PDML2: /home/cxl173430/Documents/projects/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results/
# PDML4: /home/cxl173430/Projects/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results

parser.add_argument(
    "--saved_spec_dir", default="Debug", 
    type=str, help="specific experiment path."
    )
parser.add_argument(
    '--batch_size', default=64, 
    type=int, help='The index of the gpu used'
    )
parser.add_argument(
    '--gpu', default=9, 
    type=int, help='The index of the gpu used'
    )
parser.add_argument(
    '--num_comp', default=1, 
    type=int, help='The number of composite elements'
    )

# parser.add_argument(
    # '--backbone', default='EfficientNet', 
    # type=str, help='backbone models'
    # )
parser.add_argument(
    '--seed', default=42, 
    type=int, help='set random seed'
    )

# parser.add_argument(
#     '--epochs_stage_1', default=25, 
#     type=int, help='the first stage of epochs'
#     )
# parser.add_argument(
#     '--epochs_stage_2', default=10, 
#     type=int, help='the second stage of epochs'
#     )


