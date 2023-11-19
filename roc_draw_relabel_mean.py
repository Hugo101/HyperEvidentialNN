import os
import sys
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
import copy
import yaml
import torch
torch.set_num_threads(4)

from config_args import parser  
from common_tools import create_path, set_device, dictToObj, set_random_seeds
from data.tinyImageNet import tinyImageNetVague
from data.cifar100 import CIFAR100Vague
from data.cifar10h import CIFAR10h
from data.cifar10 import CIFAR10
from data.breeds import BREEDSVague
from backbones import HENN_EfficientNet, HENN_ResNet50, HENN_VGG16, HENN_ResNet18
from backbones import EfficientNet_pretrain, ResNet50, VGG16, ResNet18
import argparse
import torch.nn as nn
import pickle as pkl

def entropy_softmax(pred):
    m = nn.Softmax(dim=1)
    prob = m(pred)
    entropy = - prob * torch.log(prob+1e-10)
    total_un = torch.sum(entropy, dim=1)
    return total_un


def entropy_SL(alpha):
    S = torch.sum(alpha, dim=1, keepdims=True)
    prob = alpha / S
    entropy = - prob * torch.log(prob+1e-10)
    entropy_s = torch.sum(entropy, dim=1)
    # entropy_m = torch.mean(entropy_s)
    return entropy_s


def vacuity_SL(alpha):
    # Vacuity uncertainty
    class_num = alpha.shape[1]
    # alpha = mean + 2.0
    S = torch.sum(alpha, dim=1, keepdims=True)
    un_vacuity = class_num / S

    return un_vacuity


def Bal(b_i, b_j):
    result = 1 - np.abs(b_i - b_j) / (b_i + b_j + 1e-7)
    return result

# def Bal(b_i, b_j):
#     result = 1 - torch.abs(b_i - b_j) / (b_i + b_j + 1e-7)
#     return result


def dissonance_SL(alpha):
    alpha = alpha.cpu().numpy()
    evidence = alpha - 1
    # alpha = mean + 2.0
    S = np.sum(alpha, axis=1, keepdims=True)
    belief = evidence / S
    dis_un = np.zeros_like(S)
    for k in range(belief.shape[0]):
        for i in range(belief.shape[1]):
            bi = belief[k][i]
            term_Bal = 0.0
            term_bj = 0.0
            for j in range(belief.shape[1]):
                if j != i:
                    bj = belief[k][j]
                    term_Bal += bj * Bal(bi, bj)
                    term_bj += bj
            dis_ki = bi * term_Bal / (term_bj + 1e-7)
            dis_un[k] += dis_ki

    return dis_un


@torch.no_grad()
def evaluate_set_DNN(model, data_loader, W, K, device):
    model.eval()
    outputs_all = []
    labels_all = [] # including composite labels

    for batch in data_loader:
        images, single_labels_GT, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        single_labels_GT = single_labels_GT.to(device, non_blocking=True)
        output = model(images)
        outputs_all.append(output)
        labels_all.append(labels)

    outputs_all = torch.cat(outputs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    uncertain = entropy_softmax(outputs_all)
    uncertain = uncertain.cpu().numpy()

    return uncertain


@torch.no_grad()
def evaluate_set_ENN(model, data_loader, W, K, device):
    model.eval()
    outputs_all = []
    labels_all = [] # including composite labels

    for batch in data_loader:
        images, single_labels_GT, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        single_labels_GT = single_labels_GT.to(device, non_blocking=True)
        output = model(images)
        outputs_all.append(output)
        labels_all.append(labels)

    outputs_all = torch.cat(outputs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    un_vacuity = vacuity_SL(outputs_all + 1)
    un_vacuity = un_vacuity.cpu().numpy()
    un_dis = dissonance_SL(outputs_all + 1)
    
    return un_vacuity, un_dis



@torch.no_grad()
def evaluate_set_HENN(model, data_loader, W, K, device):
    vaguenesses = []
    is_vague = []
    
    model.eval()
    outputs_all = []
    labels_all = [] # including composite labels

    for batch in data_loader:
        images, single_labels_GT, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        single_labels_GT = single_labels_GT.to(device, non_blocking=True)
        output = model(images)
        outputs_all.append(output)
        labels_all.append(labels)

    outputs_all = torch.cat(outputs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    # b = output / (torch.sum(output, dim=1) + W)[:, None]
    b = outputs_all / (torch.sum(outputs_all, dim=1, keepdim=True) + W)
    vaguenesses = torch.sum(b[:, K:], dim=1).cpu().numpy()
    is_vague = labels_all > K-1
    is_vague = is_vague.cpu().numpy()
    
    # for batch in data_loader:
    #     images, labels = batch
    #     images, labels = images.to(device), labels.to(device)
    #     output = model(images)
    #     b = output / (torch.sum(output, dim=1) + W)[:, None]
    #     total_vaguenesses = torch.sum(b[:, K:], dim=1)
    #     is_vague += [y >= K for y in labels.detach().cpu().numpy().tolist()]
    #     vaguenesses += total_vaguenesses.detach().cpu().numpy().tolist()
    return is_vague, vaguenesses         


def draw_roc(
    num_comp, gauss_kernel_size,
    model_HENN, model_ENN, model_DNN, 
    data_loader, 
    num_singles, num_comps,
    saved_roc_figures_dir,
    device, bestModel=True):
    # One hot encode the labels in order to plot them
    # y_onehot = pd.get_dummies(y, columns=model.classes_)

    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    # W = num_comps+num_singles
    W = num_singles
    metrics = []
    is_vague, vaguenesses = evaluate_set_HENN(model_HENN, data_loader, W, num_singles, device)
    metrics.append(vaguenesses)
    num_vague = np.sum(is_vague)
    num_single = len(is_vague) - num_vague
    print(f"The num of Singletons and Composites: {num_single, num_vague}: {num_vague/(num_single+num_vague):.2f}")
    
    vacuity_ENN, diss_ENN = evaluate_set_ENN(model_ENN, data_loader, W, num_singles, device)
    metrics.append(vacuity_ENN)
    metrics.append(diss_ENN)
    
    entropy_DNN = evaluate_set_DNN(model_DNN, data_loader, W, num_singles, device)
    metrics.append(entropy_DNN)

    # tag = ["Vagueness", "Vacuity", "Dissonance", "Entropy"]
    return metrics, is_vague

def make(args):
    mydata = None
    num_singles = 0
    num_comps = 0
    num_classes_both = 0 
    
    if args.dataset == "tinyimagenet":
        mydata = tinyImageNetVague(
            args.data_dir, 
            num_comp=args.num_comp, 
            batch_size=args.batch_size,
            imagenet_hierarchy_path=args.data_dir,
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
            blur=args.blur,
            gauss_kernel_size=args.gauss_kernel_size,
            pretrain=args.pretrain,
            num_workers=args.num_workers,
            seed=args.seed,
            comp_el_size=args.num_subclasses,
            )
    elif args.dataset == "CIFAR10h":
        mydata = CIFAR10h(
            args.data_dir,
            batch_size=args.batch_size,
            duplicate=True,
            pretrain=args.pretrain,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    elif args.dataset == "CIFAR10":
        mydata = CIFAR10(
            args.data_dir,
            batch_size=args.batch_size,
            pretrain=args.pretrain,
            num_workers=args.num_workers,
            seed=args.seed,
        )

    num_singles = mydata.num_classes
    num_comps = mydata.num_comp
    print(f"Data: {args.dataset}, num of singleton and composite classes: {num_singles, num_comps}")
    
    num_classes_both = num_singles + num_comps
    if args.backbone == "EfficientNet-b3":
        model_HENN = HENN_EfficientNet(num_classes_both, pretrain=args.pretrain)
        model_ENN = HENN_EfficientNet(num_singles, pretrain=args.pretrain)
        model_DNN = EfficientNet_pretrain(num_singles, pretrain=args.pretrain)
    elif args.backbone == "ResNet50":
        model_HENN = HENN_ResNet50(num_classes_both)
        model_ENN = HENN_ResNet50(num_singles)
        model_DNN = ResNet50(num_singles)
    elif args.backbone == "VGG16":
        model_HENN = HENN_VGG16(num_classes_both)
        model_ENN = HENN_VGG16(num_singles)
        model_DNN = VGG16(num_singles)
    elif args.backbone == "ResNet18":
        model_HENN = HENN_ResNet18(num_classes_both, pretrain=args.pretrain)
        model_ENN = HENN_ResNet18(num_singles, pretrain=args.pretrain)
        model_DNN = ResNet18(num_singles, pretrain=args.pretrain)
    else:
        print(f"### ERROR {args.dataset}: The backbone {args.backbone} is invalid!")

    model_HENN = model_HENN.to(args.device)
    model_ENN = model_ENN.to(args.device)
    model_DNN = model_DNN.to(args.device)

    return mydata, model_HENN, model_ENN, model_DNN


def main(args):
    set_random_seeds(args.seed)
    device = args.device
    mydata, model_HENN, model_ENN, model_DNN = make(args)
    num_singles = mydata.num_classes
    num_comps = mydata.num_comp

    metrics = []
    # for loop for different models
    # seeds = [42, 18, 86]
    seeds = [42, 18, 86, 98, 60]
    for seed in seeds:
        # # for dataset CIFAR10h-relabeled
        # spec_HENN = f"SEED{seed}_BBResNet18_5M_Ker11_sweep_GDDexp101/lr_0.0001_klLamGDD_0.0_EntrLamDir_0.0_EntrLamGDD_1"
        # spec_ENN = f"5M_ker11_Seed{seed}_BBResNet18_sweep_ENN/lr0.001_EntrLam0.01"
        # spec_DNN = f"5M_ker11_Seed{seed}_sweep_DNN/0.001"
        
        # # for dataset CIFAR10-relabeled
        # spec_HENN = f"SEED{seed}_BBResNet18_5M_Ker11_sweep_GDDexp101/lr_0.001_klLamGDD_0.0_EntrLamDir_0.0_EntrLamGDD_1"
        # spec_ENN = f"5M_ker11_Seed{seed}_BBResNet18_sweep_ENN/lr0.001_EntrLam0.01"
        # spec_DNN = f"5M_ker11_Seed{seed}_BBResNet18_sweep_DNN/0.001"
        
        # for dataset CIFAR10-relabeled pretrained Model
        spec_HENN = f"SEED{seed}_BBEfficientNet-b3_5M_Ker11_sweep_GDDexp101/lr_1e-05_klLamGDD_0.0_EntrLamDir_0.0_EntrLamGDD_1"
        spec_ENN = f"5M_ker11_Seed{seed}_BBEfficientNet-b3_sweep_ENN/lr0.0001_EntrLam0.01"
        spec_DNN = f"5M_ker11_Seed{seed}_BBEfficientNet-b3_sweep_DNN/1e-05"
        saved_path_HENN = os.path.join(args.base_path_spec_HENN, spec_HENN, "model_uncertainty_gdd.pt")
        saved_path_ENN = os.path.join(args.base_path_spec_ENN, spec_ENN, "model_uncertainty_digamma.pt")
        saved_path_DNN = os.path.join(args.base_path_spec_DNN, spec_DNN, "model_CrossEntropy.pt")
        
        checkpoint = torch.load(saved_path_HENN, map_location=device)
        model_HENN.load_state_dict(checkpoint["model_state_dict_best"])
        print(f"\n### HENN Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:")
        
        checkpoint = torch.load(saved_path_ENN, map_location=device)
        model_ENN.load_state_dict(checkpoint["model_state_dict_best"])
        print(f"\n### ENN Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:")
        
        checkpoint = torch.load(saved_path_DNN, map_location=device)
        model_DNN.load_state_dict(checkpoint["model_state_dict_best"])
        print(f"\n### DNN Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:")
        
        metric, is_vague = draw_roc(
            args.num_comp, args.gauss_kernel_size,
            model_HENN, model_ENN, model_DNN, 
            mydata.test_loader, 
            num_singles, num_comps,
            args.saved_roc_figures_dir,
            device, bestModel=True)
        print("*** Each metric for the current seed:", [ele.shape for ele in metric])
        metrics.append(metric)
    metrics.append(is_vague)
    
    # save the metrics using pickle
    save_path = os.path.join(f"{args.dataset}_metrics.pkl")
    with open(save_path, "wb") as f:
        pkl.dump(metrics, f)
    print(f"### Save the metrics to {save_path}")
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Conformalize Torchvision Model')
    parser.add_argument('--data_dir', default="/home/cxl173430/data/DATASETS/", type=str, help='path to dataset')
    parser.add_argument(
        "--output_folder", 
        default="/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results/", 
        type=str, help="where results will be saved."
    )
    # parser.add_argument(
    #     "--saved_spec_dir", default="CIFAR100/Statistics", 
    #     type=str, help="specific experiment path."
    #     )
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset name')
    parser.add_argument('--gauss_kernel_size', default=7, type=int, help='gaussian kernel size')
    parser.add_argument('--num_comp', default=15, type=int, help='number of composite classes')
    
    args = parser.parse_args()
    opt = vars(args)

    # build the path to save model and results
    if args.dataset == "tinyimagenet":
        saved_spec_dir = f"Tiny/Statistics"
    elif args.dataset == "cifar100":
        saved_spec_dir = f"CIFAR100/Statistics"
    elif args.dataset == "CIFAR10h":
        saved_spec_dir = f"CIFAR10h/Statistics"
    elif args.dataset == "CIFAR10":
        saved_spec_dir = f"CIFAR10/Statistics"
    base_path = os.path.join(args.output_folder, saved_spec_dir)
    saved_path = os.path.join(base_path, "ROC_figures")
    config_file = os.path.join(saved_path, "config.yml")
    CONFIG = yaml.load(open(config_file), Loader=yaml.FullLoader)
    opt.update(CONFIG)
    opt["device"] = set_device(args.gpu)
    
    opt["saved_roc_figures_dir"] = saved_path
    
    if args.dataset == "tinyimagenet":
        spec_dir = f"20M_15M_10M_357ker_sweep_HENNexp5_pytorchKer/{args.num_comp}M_ker{args.gauss_kernel_size}_sweep_HENNexp5/lr_1e-05_EntropyLam_0.1"
        opt["base_path_spec_HENN"] = os.path.join(base_path, spec_dir)
        
        spec_dir = f"20M_15M_10M_357ker_sweep_ENN_pytorchKer_UCE/{args.num_comp}M_ker{args.gauss_kernel_size}_sweep_ENN/1e-05"
        opt["base_path_spec_ENN"] = os.path.join(base_path, spec_dir)
        
        spec_dir = f"20M_15M_10M_357ker_sweep_DNN_pytorchKer/{args.num_comp}M_ker{args.gauss_kernel_size}_sweep_DNN/1e-05"
        opt["base_path_spec_DNN"] = os.path.join(base_path, spec_dir)
    elif args.dataset == "cifar100":
        spec_dir = f"20M_15M_10M_357ker_sweep_HENNexp5_pytorchKer_V2/{args.num_comp}M_ker{args.gauss_kernel_size}_sweep_HENNexp5/lr_1e-05_EntropyLam_0.1"
        opt["base_path_spec_HENN"] = os.path.join(base_path, spec_dir)
        
        spec_dir = f"20M_15M_10M_357ker_sweep_ENN_pytorchKer_V2_UCE/{args.num_comp}M_ker{args.gauss_kernel_size}_sweep_ENN/1e-05"
        opt["base_path_spec_ENN"] = os.path.join(base_path, spec_dir)
        
        spec_dir = f"20M_15M_10M_357ker_sweep_DNN_pytorchKer_V2/{args.num_comp}M_ker{args.gauss_kernel_size}_sweep_DNN/1e-05"
        opt["base_path_spec_DNN"] = os.path.join(base_path, spec_dir)
    
    elif args.dataset == "CIFAR10h":
        # spec_dir = "sweep_GDD_klGDD_1025/SEED42_BBEfficientNet-b3_5M_Ker11_sweep_GDDexp101/lr_0.0001_klLamGDD_0.01_EntrLamDir_0.0_EntrLamGDD_0.0"
        # spec_dir = "sweep_GDD_klGDD_1025/SEED42_BBEfficientNet-b3_5M_Ker11_sweep_GDDexp101/lr_1e-05_klLamGDD_0.01_EntrLamDir_0.0_EntrLamGDD_0.0"
        # spec_dir = "sweep_GDD_klGDD_1025_pretrainFalse/SEED42_5M_Ker11_sweep_GDDexp101/lr_0.0001_klLamGDD_1_EntrLamDir_0.0_EntrLamGDD_0.0"
        # spec_dir = "SEED60_BBResNet18_5M_Ker11_sweep_GDDexp101/lr_0.001_klLamGDD_1_EntrLamDir_0.0_EntrLamGDD_0.0" # good
        # spec_dir = "SEED42_BBResNet18_5M_Ker11_sweep_GDDexp101/lr_0.001_klLamGDD_1_EntrLamDir_0.0_EntrLamGDD_0.0" #good
        # spec_dir = "sweep_GDD_GDDentr_1105_pretrainFalse/SEED86_BBResNet18_5M_Ker11_sweep_GDDexp101/lr_0.001_klLamGDD_0.0_EntrLamDir_0.0_EntrLamGDD_1"  # good
        # spec_dir = "sweep_GDD_GDDentr_1105_pretrainFalse/SEED60_BBResNet18_5M_Ker11_sweep_GDDexp101/lr_0.0001_klLamGDD_0.0_EntrLamDir_0.0_EntrLamGDD_1" # good
        # spec_dir = "sweep_GDD_GDDentr_1105_pretrainFalse/SEED42_BBResNet18_5M_Ker11_sweep_GDDexp101/lr_0.0001_klLamGDD_0.0_EntrLamDir_0.0_EntrLamGDD_1"  # good
        spec_dir = "sweep_GDD_GDDentr_1105_pretrainFalse" 
        
        opt["base_path_spec_HENN"] = os.path.join(base_path, spec_dir)
        
        # spec_dir = "sweep_ENN_1031/5M_ker11_Seed42_BBEfficientNet-b3_sweep_ENN/lr0.0001_EntrLam0.001"
        spec_dir = "sweep_ENN_1031_pretrainFalse"
        # spec_dir = "sweep_ENN_1031_pretrainFalse/5M_ker11_Seed42_BBResNet18_sweep_ENN/lr0.001_EntrLam0"
        opt["base_path_spec_ENN"] = os.path.join(base_path, spec_dir)
        
        # spec_dir = "sweep_DNN_1030/5M_ker11_Seed42_BBEfficientNet-b3_sweep_DNN/1e-05"
        spec_dir = "sweep_DNN_1030_pretrainFalse"
        opt["base_path_spec_DNN"] = os.path.join(base_path, spec_dir)
    
    elif args.dataset == "CIFAR10":
        spec_dir = "sweep_GDD_GDDentr_1114" 
        # spec_dir = "sweep_GDD_GDDentr_1114_pretrainFalse" 
        opt["base_path_spec_HENN"] = os.path.join(base_path, spec_dir)
        
        spec_dir = "sweep_ENN_1116"
        # spec_dir = "sweep_ENN_1114_pretrainFalse"
        opt["base_path_spec_ENN"] = os.path.join(base_path, spec_dir)
        
        spec_dir = "sweep_DNN_1115"
        # spec_dir = "sweep_DNN_1114_pretrainFalse"
        opt["base_path_spec_DNN"] = os.path.join(base_path, spec_dir)
    
    # convert args from Dict to Object
    args = dictToObj(opt)
    
    main(args)