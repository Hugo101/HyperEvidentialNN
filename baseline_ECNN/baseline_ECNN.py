import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime as dt
import time
import yaml
import random
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

# Import other standard packages

import wandb
from collections import Counter
import pandas as pd
import random
from random import randint
from sklearn import metrics
import math

import torch
import torch.backends.cudnn as cudnn
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data import Subset, ConcatDataset
from torch.utils.data import random_split, sampler 
import torchvision
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets, utils 
from efficientnet_pytorch import EfficientNet

from scipy.optimize import minimize

# DS related 
from dst_pytorch import EfficientNet_DS

data_path = "/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/"
sys.path.insert(1, data_path)
from backbones import EfficientNet_pretrain, ResNet50
from data.tinyImageNet import tinyImageNetVague
from data.cifar100 import CIFAR100Vague
from common_tools import set_device, create_path, dictToObj, set_random_seeds
from helper_functions import ReduceLabelDataset

parser = argparse.ArgumentParser(description='Conformalize Torchvision Model')
parser.add_argument('--data_dir', default="/home/cxl173430/data/DATASETS/", type=str, help='path to dataset')
parser.add_argument(
    "--output_folder", 
    default="/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results/", 
    type=str, help="where results will be saved."
)
parser.add_argument(
    "--saved_spec_dir", default="HyperEvidentialNN/baseline_RAPS", 
    type=str, help="specific experiment path."
    )
parser.add_argument('--gpu', default=0, type=int, help='GPU ID')
parser.add_argument('--seed', default=42, type=int, help='random seed')

args = parser.parse_args()
opt = vars(args)
# build the path to save model and results
create_path(args.output_folder) 
base_path = os.path.join(args.output_folder, args.saved_spec_dir)
create_path(base_path)
config_file = os.path.join(base_path, "config.yml")
config = yaml.load(open(config_file), Loader=yaml.FullLoader)
opt.update(config)
args = opt
args = dictToObj(args)

device = set_device(args.gpu)

def data_log(
    prec_recall_f, 
    comp_GT_cnt, cmp_pred_cnt, 
    js_result, 
    accs, 
    bestModel=True):
    if bestModel:
        tag = "TestB"
    else:
        tag = "TestF"
    
    wandb.log({
        f"{tag} JSoverall": js_result[0], 
        f"{tag} JScomp": js_result[1], 
        f"{tag} JSsngl": js_result[2],
        f"{tag} CmpPreci": prec_recall_f[0], 
        f"{tag} CmpRecal": prec_recall_f[1], 
        f"{tag} CmpFscor": prec_recall_f[2], 
        f"{tag} NonVagueAcc0": accs[0],
        f"{tag} NonVagueAcc1": accs[1],
        f"{tag} NonVagueAcc2": accs[2],
        f"{tag} comp_GT_cnt": comp_GT_cnt,
        f"{tag} comp_pred_cnt": cmp_pred_cnt,
        }, step=None)
    print(f"{tag} NonVagueAcc: {accs[0]:.4f}, {accs[1]:.4f}, {accs[2]:.4f}, \n \
            JS(O_V_N): {js_result[0]:.4f}, {js_result[1]:.4f}, {js_result[2]:.4f}, \n \
            P_R_F_compGTcnt_cmpPREDcnt: {prec_recall_f}\n")


def make(args):
    mydata = None
    num_singles = 0

    if args.dataset == "tinyimagenet":
        mydata = tinyImageNetVague(
                    args.data_dir, 
                    num_comp=args.num_comp, 
                    batch_size=args.batch_size,
                    imagenet_hierarchy_path=args.data_dir,
                    duplicate=True)
    elif args.dataset == "cifar100":
        mydata = CIFAR100Vague(
                    args.data_dir, 
                    num_comp=args.num_comp,
                    batch_size=args.batch_size,
                    imagenet_hierarchy_path=args.data_dir,
                    duplicate=True)
    print(f"Size of training/validation/test:")
    print(len(mydata.train_loader.dataset)) # >90200 because of the duplicates
    print(len(mydata.valid_loader.dataset))
    print(len(mydata.test_loader.dataset))

    num_singles = mydata.num_classes
    num_comps = mydata.num_comp
    print(f"Data: {args.dataset}, num of singleton and composite classes: {num_singles, num_comps}")

    # duplicate validation set
    valid_ds = mydata.modify_vague_samples(mydata.valid_loader.dataset)
    # reduce one additional label info for validation set
    valid_ds_reducedLabel = ReduceLabelDataset(valid_ds)
    valid_loader = DataLoader(valid_ds_reducedLabel, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    assert mydata.valid_loader.dataset.dataset is not None
    
    # define pretrained CNN model
    num_singles = mydata.num_classes
    if args.backbone == "EfficientNet-b3":
        model = EfficientNet_pretrain(num_singles)
    elif args.backbone == "ResNet50":
        model = ResNet50(num_singles)
    model = model.to(device)

    if args.backbone == "EfficientNet-b3":
        model_DS = EfficientNet_DS(num_singles)
    elif args.backbone == "ResNet50":
        model_DS = ResNet50(num_singles)
    model_DS = model_DS.to(device)
    
    return mydata, valid_loader, model


def train_ds_model():
    
    return

def test_ds_model():
    
    
    return

def main():
    ## Fix randomness
    set_random_seeds(seed=args.seed) # 42
    mydata, valid_loader, model = make(args)

    saved_spec_dir_DNN = args.saved_spec_dir_DNN
    model_saved_base_path = os.path.join(args.output_folder, saved_spec_dir_DNN)
    print("DNN model saved path:", model_saved_base_path)
    saved_path = os.path.join(model_saved_base_path, "model_CrossEntropy.pt")
    # load pretrained CNN model
    checkpoint = torch.load(saved_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    raps(model, mydata, valid_loader, bestModel=False)
    
    model_best_from_valid = copy.deepcopy(model)
    model_best_from_valid.load_state_dict(checkpoint["model_state_dict_best"]) 
    raps(model_best_from_valid, mydata, valid_loader, bestModel=True)


if __name__ == "__main__":
    # tell wandb to get started
    print(config)
    with wandb.init(project=f"{config['dataset']}-{config['num_comp']}M-RAPS", config=config):
        config = wandb.config
        main()