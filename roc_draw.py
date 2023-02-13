import os
import sys
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import copy
import yaml
import torch
torch.set_num_threads(4)

from config_args import parser  
from common_tools import create_path, set_device, dictToObj, set_random_seeds
from data.tinyImageNet import tinyImageNetVague
from data.cifar100 import CIFAR100Vague
from data.breeds import BREEDSVague
from backbones import HENN_EfficientNet, HENN_ResNet50, HENN_VGG16
from backbones import EfficientNet_pretrain, ResNet50, VGG16
import argparse

@torch.no_grad()
def evaluate_set_DNN(model, data_loader, W, K, device):
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

    return is_vague, vaguenesses     


@torch.no_grad()
def evaluate_set_ENN(model, data_loader, W, K, device):
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

    return is_vague, vaguenesses     



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
    model_HENN, model_DNN, model_ENN, 
    data_loader, 
    num_singles, num_comps, 
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
    W = num_comps+num_singles
    metrics = []
    is_vague, vaguenesses = evaluate_set_HENN(model_HENN, data_loader, W, num_singles, device)
    metrics.append(vaguenesses)
    _, entropy_DNN = evaluate_set_DNN(model_DNN, data_loader, W, num_singles, device)
    metrics.append(entropy_DNN)
    _, entropy_ENN = evaluate_set_ENN(model_ENN, data_loader, W, num_singles, device)
    metrics.append(entropy_ENN)
    for i in range(3):
        fpr, tpr, _ = roc_curve(is_vague, metrics[i])
        auc_score = roc_auc_score(is_vague, metrics[i])

        # name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        name = f"(AUC={auc_score})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
    
    fig.update_layout(
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=700, height=500
    )
    fig.show()
    

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

    saved_path_HENN = os.path.join(args.base_path_spec_HENN, "model_uncertainty_digamma.pt")
    saved_path_ENN = os.path.join(args.base_path_spec_ENN, "model_uncertainty_digamma.pt")
    saved_path_DNN = os.path.join(args.base_path_spec_DNN, "model_CrossEntropy.pt")
    checkpoint = torch.load(saved_path_HENN, map_location=device)
    model_HENN.load_state_dict(checkpoint["model_state_dict_best"])

    checkpoint = torch.load(saved_path_ENN, map_location=device)
    model_ENN.load_state_dict(checkpoint["model_state_dict_best"])
    
    checkpoint = torch.load(saved_path_DNN, map_location=device)
    model_DNN.load_state_dict(checkpoint["model_state_dict_best"])
    
    print(f"\n### Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:")
    draw_roc(model_HENN, model_ENN, model_DNN, mydata.test_loader, num_singles, num_comps, device, bestModel=True)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Conformalize Torchvision Model')
    parser.add_argument('--data_dir', default="/home/cxl173430/data/DATASETS/", type=str, help='path to dataset')
    parser.add_argument(
        "--output_folder", 
        default="/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results/", 
        type=str, help="where results will be saved."
    )
    parser.add_argument(
        "--saved_spec_dir", default="Tiny/15M_ker5_sweep_HENNexp5_pytorchKer", 
        type=str, help="specific experiment path."
        )
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    args = parser.parse_args()
    opt = vars(args)

    # build the path to save model and results
    base_path = os.path.join(args.output_folder, args.saved_spec_dir)
    config_file = os.path.join(base_path, "config.yml")
    CONFIG = yaml.load(open(config_file), Loader=yaml.FullLoader)
    opt.update(CONFIG)
    opt["device"] = set_device(args.gpu)
    
    spec_dir = "lr_1e-05_EntropyLam_0.1"
    opt["base_path_spec_HENN"] = os.path.join(base_path, spec_dir)
    
    spec_dir = "lr_1e-05_EntropyLam_0.1"
    opt["base_path_spec_HENN"] = os.path.join(base_path, spec_dir)
    
    spec_dir = "lr_1e-05_EntropyLam_0.1"
    opt["base_path_spec_HENN"] = os.path.join(base_path, spec_dir)
    
    # convert args from Dict to Object
    args = dictToObj(opt)
    
    main(args)