# Import libraries
import os, sys, inspect
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import yaml
import random
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

# Import other standard packages
import torch
torch.set_num_threads(4)
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.backends.cudnn as cudnn
import torchvision
import wandb
from collections import Counter

from conformal_classification.conformal import *
from conformal_classification.utils import *

data_path = "/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/"
sys.path.insert(1, data_path)
from backbones import EfficientNet_pretrain, ResNet50, VGG16
from data.tinyImageNet import tinyImageNetVague
from data.cifar100 import CIFAR100Vague
from data.breeds import BREEDSVague
from common_tools import set_device, create_path, dictToObj, set_random_seeds
from helper_functions import AddLabelDataset

parser = argparse.ArgumentParser(description='Conformalize Torchvision Model')
parser.add_argument('--data_dir', default="/home/cxl173430/data/DATASETS/", type=str, help='path to dataset')
parser.add_argument(
    "--output_folder", 
    default="/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results/", 
    type=str, help="where results will be saved."
)
parser.add_argument(
    "--saved_spec_dir", default="CIFAR100/20M_15M_10M_357ker_RAPS/15M_ker3_sweep_DNN_pytorchKer", 
    type=str, help="specific experiment path."
    )
parser.add_argument('--gpu', default=0, type=int, help='GPU ID')
parser.add_argument('--kreg', default=-1.0, type=float, help='kreg')
parser.add_argument('--lamda', default=-1.0, type=float, help='lambda')
parser.add_argument('--alpha', default=0.1, type=float, help='alpha')
# parser.add_argument('--num_workers', metavar='NW', help='number of workers', default=0)
# parser.add_argument('--num_calib', metavar='NCALIB', help='number of calibration points', default=10000)
parser.add_argument('--seed', default=42, type=int, help='random seed')

args = parser.parse_args()
opt = vars(args)
# build the path to save model and results
create_path(args.output_folder) 
base_path = os.path.join(args.output_folder, args.saved_spec_dir)
create_path(base_path)
config_file = os.path.join(base_path, "config_raps.yml")
config = yaml.load(open(config_file), Loader=yaml.FullLoader)
opt.update(config)
args = opt
args = dictToObj(args)
device = set_device(args.gpu)



def precision_recall_f(y_test, y_pred):
    # make singleton labels 0, and composite labels 1
    y_test = y_test.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    label_value_cnt = Counter(y_test)
    pred_value_cnt = Counter(y_pred)

    comp_GT_cnt = label_value_cnt[True]
    cmp_pred_cnt = pred_value_cnt[True]

    return precision, recall, f1, comp_GT_cnt, cmp_pred_cnt


def data_log(
    prec_recall_f, 
    comp_GT_cnt, cmp_pred_cnt, 
    js_result, 
    accs, 
    valid_test="valid"):

    tag = valid_test
    
    wandb.log({
        f"{tag}_JSoverall": js_result[0], 
        f"{tag}_JScomp": js_result[1], 
        f"{tag}_JSsngl": js_result[2],
        f"{tag}_CmpPreci": prec_recall_f[0], 
        f"{tag}_CmpRecal": prec_recall_f[1], 
        f"{tag}_CmpFscor": prec_recall_f[2], 
        f"{tag}_NonVagueAcc0": accs[0],
        f"{tag}_NonVagueAcc1": accs[1],
        f"{tag}_NonVagueAcc2": accs[2],
        f"{tag}_comp_GT_cnt": comp_GT_cnt,
        f"{tag}_comp_pred_cnt": cmp_pred_cnt,
        }, step=None)
    print(f"{tag}_NonVagueAcc: {accs[0]:.4f}, {accs[1]:.4f}, {accs[2]:.4f}, \n \
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
                    duplicate=True,
                    blur=args.blur,
                    gray=args.gray,
                    gauss_kernel_size=args.gauss_kernel_size,
                    pretrain=args.pretrain,
                    num_workers=args.num_workers,
                    seed=args.seed)

    elif args.dataset == "cifar100":
        mydata = CIFAR100Vague(
                    args.data_dir, 
                    num_comp=args.num_comp,
                    batch_size=args.batch_size,
                    duplicate=True,
                    blur=args.blur,
                    gauss_kernel_size=args.gauss_kernel_size,
                    pretrain=args.pretrain,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    comp_el_size=args.num_subclasses,
                    )

    elif args.dataset in ["living17", "nonliving26", "entity13", "entity30"]:
        data_path_base = os.path.join(args.data_dir, "ILSVRC/ILSVRC")
        mydata = BREEDSVague(
            os.path.join(data_path_base, "BREEDS/"),
            os.path.join(data_path_base, 'Data', 'CLS-LOC/'),
            ds_name=args.dataset,
            num_comp=args.num_comp, 
            batch_size=args.batch_size,
            duplicate=True,  #key duplicate
            blur=args.blur,
            gauss_kernel_size=args.gauss_kernel_size,
            pretrain=args.pretrain,
            num_workers=args.num_workers,
            seed=args.seed,
            comp_el_size=args.num_subclasses,
            )
    
    print(f"Size of training/validation/test:")
    print(len(mydata.train_loader.dataset)) # >90200 because of the duplicates
    print(len(mydata.valid_loader.dataset))
    print(len(mydata.test_loader.dataset))

    num_singles = mydata.num_classes
    num_comps = mydata.num_comp
    print(f"Data: {args.dataset}, num of singleton and composite classes: {num_singles, num_comps}")

    # duplicate validation set
    # valid_ds = mydata.modify_vague_samples(mydata.valid_loader.dataset)
    valid_ds = mydata.valid_loader.dataset
    # reduce one additional label info for validation set
    valid_ds_reducedLabel = ReduceLabelDataset(valid_ds)

    # assert mydata.valid_loader.dataset.dataset is not None
    
    # define pretrained CNN model
    num_singles = mydata.num_classes
    if args.backbone == "EfficientNet-b3":
        model = EfficientNet_pretrain(num_singles)
    elif args.backbone == "ResNet50":
        model = ResNet50(num_singles)
    elif args.backbone == "VGG16":
        model = VGG16(num_singles)
    model = model.to(device)

    return mydata, valid_ds_reducedLabel, model


def evaluate_RAPS(cmodel, mydata, test_ds, valid_test="valid"):
    correct_vague = 0.0
    correct_nonvague = 0.0
    vague_total = 0
    nonvague_total = 0

    
    R = mydata.R

    #vague prediction
    pred_sets = []
    ground_sets = []

    #nonvague prediction
    num_corr_1 = 0 # use original label before blurring
    num_corr_2 = 0 # if nonvague prediction is in composite labels, then +1
    num_corr_3 = 0 # similiar to JS

    comp_or_not_original = []
    comp_or_not_pred = []

    if mydata.name == "tinyimagenet":
        height = 224
    elif mydata.name == "cifar100":
        height = 224
        
    for data in test_ds:
        img, label_singl, label = data
        scores, set_pred_array = cmodel(img.view(1,3,height,height).to(device))
        set_predict = {s for s in set_pred_array[0]}
        set_ground = {s for s in R[label]}
        
        pred_sets.append(set_predict)
        ground_sets.append(set_ground)
            
        inter_set = set_predict.intersection(set_ground)
        union_set = set_predict.union(set_ground)
        print("\n")
        # print("preded   set: ", set_predict)
        # print("ground truth: ", set_ground)
        # print("inter    set: ", inter_set)
        # print("union    set: ", union_set)

        rate = float(len(inter_set)/len(union_set))
        if len(set_predict) == 1:
            correct_nonvague += rate
            nonvague_total += 1
            comp_or_not_pred.append(0)
        elif len(set_predict) > 1:
            correct_vague += rate 
            vague_total += 1
            comp_or_not_pred.append(1)
        else:
            print(f"ERROR: {len(set_predict)} should be positive")
        
        if len(set_ground) == 1:
            comp_or_not_original.append(0)
        elif len(set_ground) > 1:
            comp_or_not_original.append(1)
        else:
            print(f"ERROR: {len(set_ground)} should be positive")
            
        ### nonvague prediction
        pred_singl = torch.argmax(scores, dim=1).cpu().item()
        # 1.
        if pred_singl == label_singl:
            num_corr_1 += 1
        # 2.
        if pred_singl in set_ground:
            num_corr_2 += 1
        # 3.
        inter_tmp = set_ground.intersection(set([pred_singl]))
        union_tmp = set_ground.union(set([pred_singl]))
        num_corr_3 += float(len(inter_tmp)/len(union_tmp))
        ### END
    #     break
    # print(pred_singl, label_singl, num_corr_1)
    # print(pred_singl, set_ground, num_corr_2)
    # print(pred_singl, set_ground, num_corr_3)

    comp_or_not_original = torch.tensor(comp_or_not_original)
    comp_or_not_pred = torch.tensor(comp_or_not_pred)
    precision, recall, f1, comp_GT_cnt, cmp_pred_cnt = precision_recall_f(comp_or_not_original, comp_or_not_pred)
    prec_recall_f = [precision, recall, f1]

    stat_result=[correct_nonvague, correct_vague, nonvague_total, vague_total]
    print(f"Count for JS: {stat_result}")
    js_nonvague = stat_result[0] / (stat_result[2]+1e-10)
    js_vague = stat_result[1] / (stat_result[3]+1e-10)
    js_overall = (stat_result[0] + stat_result[1])/(stat_result[2] + stat_result[3]+1e-10)
    js_result = [js_overall, js_vague, js_nonvague]

    acc_1 = num_corr_1 / len(test_ds)
    acc_2 = num_corr_2 / len(test_ds)
    acc_3 = num_corr_3 / len(test_ds)
    accs = [acc_1, acc_2, acc_3]
    data_log(prec_recall_f, comp_GT_cnt, cmp_pred_cnt, js_result, accs, valid_test=valid_test)


def raps(
    model, mydata, calib_valid_data, 
    kreg=None, lamda=None,
    alpha=0.1,
    bestModel=True):
    # RAPS
    if kreg == -1.0:
        kreg = None
    if lamda == -1.0:
        lamda = None
        
    cudnn.benchmark = True
    # Get your model
    _ = model.eval()
    model.to(device)
    
    # divide validation data into calib_data and valid_data
    total = len(calib_valid_data)
    
    num_val = 1000
    calib_data, valid_data = torch.utils.data.random_split(calib_valid_data, [total-num_val, num_val])

    # Initialize loaders 
    calib_loader = torch.utils.data.DataLoader(calib_data, batch_size=64, shuffle=True, pin_memory=True)


    # Conformalize model
    cmodel = ConformalModel(
        model, calib_loader, mydata.name, alpha=alpha, 
        kreg=kreg, lamda=lamda,
        lamda_criterion='size')

    # validation set to select hyperparameters, such as alpha, kreg, and lambda
    valid_data = AddLabelDataset(valid_data)
    evaluate_RAPS(cmodel, mydata, valid_data, valid_test="valid")
    
    # test set for final results
    test_ds = mydata.test_loader.dataset
    evaluate_RAPS(cmodel, mydata, test_ds, valid_test="test")


def main():
    ## Fix randomness
    set_random_seeds(seed=args.seed) # 42
    mydata, valid_data_one_label, model = make(args)

    saved_spec_dir_DNN = args.saved_spec_dir_DNN
    model_saved_base_path = os.path.join(base_path, saved_spec_dir_DNN)
    print("DNN model saved path:", model_saved_base_path)
    saved_path = os.path.join(model_saved_base_path, "model_CrossEntropy.pt")
    # load pretrained CNN model
    checkpoint = torch.load(saved_path, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # raps(model, mydata, valid_loader, kreg=None, lamda=None, bestModel=False)
    
    model_best_from_valid = copy.deepcopy(model)
    model_best_from_valid.load_state_dict(checkpoint["model_state_dict_best"]) 
    raps(
        model_best_from_valid, mydata, valid_data_one_label, 
        kreg=args.kreg, lamda=args.lamda, 
        alpha=args.alpha, bestModel=True)


if __name__ == "__main__":
    # tell wandb to get started
    print(config)
    with wandb.init(project=f"{config['dataset']}-{config['backbone']}-20M-15M-10M-RAPS", config=config):
        config = wandb.config
        main()