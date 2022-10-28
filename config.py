import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--data_dir', default='./', type=str, help='dataset directory')
parser.add_argument('--gpu', default=9, type=int, help='The index of the gpu used')
parser.add_argument('--num_comp', default=5, type=int,
                    help='The number of composite elements')
parser.add_argument('--save_dir', default='./M5', type=str, help='set save directory')
parser.add_argument('--backbone', default='EfficientNet', type=str, help='backbone models')
parser.add_argument('--seed', default=123, type=int, help='set random seed')

parser.add_argument('--epochs_stage_1', default=25, type=int, help='the first stage of epochs')
parser.add_argument('--epochs_stage_2', default=10, type=int, help='the second stage of epochs')


