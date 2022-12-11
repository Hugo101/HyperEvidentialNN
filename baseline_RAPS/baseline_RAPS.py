# Import libraries
import os, sys, inspect
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from sklearn.metrics import precision_score, recall_score, f1_score

# Import other standard packages
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.backends.cudnn as cudnn
import torchvision

# sys.path.insert(1, os.path.join(sys.path[0], './conformal_classification/'))
# from conformal import *
# from utils import *

from conformal_classification.conformal import *
from conformal_classification.utils import *


data_path = "/data/cxl173430/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/"
sys.path.insert(1, data_path)
from backbones import EfficientNet_pretrain
from data.tinyImageNet import tinyImageNetVague 

seed = 42
set_random_seeds(seed)
data_dir = '/home/cxl173430/data/DATASETS/'
batch_size = 64

mydata = tinyImageNetVague(
            data_dir, 
            num_comp=1, 
            batch_size=batch_size,
            imagenet_hierarchy_path=data_dir,
            duplicate=True)

print(len(mydata.train_loader.dataset)) # >90200 because of the duplicates
print(len(mydata.valid_loader.dataset))
print(len(mydata.test_loader.dataset))

# duplicate validation set
valid_ds = mydata.modify_vague_samples(mydata.valid_loader.dataset)
valid_ds_reducedLabel = ReduceLabelDataset(valid_ds)
valid_loader = DataLoader(valid_ds_reducedLabel, batch_size=64, num_workers=1, pin_memory=True)

assert mydata.valid_loader.dataset.dataset is not None

# load pretrained CNN model
num_singles = 200
output_folder="/data/cxl173430/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results/"
# saved pretrained model 
saved_spec_dir = "tiny_baseline_DNN"
device = "cuda:0"

model = EfficientNet_pretrain(num_singles)
base_path = os.path.join(output_folder, saved_spec_dir)
print("base path:" base_path)

R = mydata.R
saved_path = os.path.join(base_path, "model_CrossEntropy.pt")
checkpoint = torch.load(saved_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model_best_from_valid = copy.deepcopy(model)
model_best_from_valid.load_state_dict(checkpoint["model_state_dict_best"]) 

# # model after the final epoch
# if args.vaguePred:
#     js_result, prec_recall_f = evaluate_vague_final(model, test_loader, valid_loader, R, device)
# if args.nonVaguePred:
#     acc_nonvague = evaluate_nonvague_final(model, test_loader, device)
# test_result_log(js_result, prec_recall_f, acc_nonvague, False) # bestModel=False

# print(f"### Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:\n")
# if args.vaguePred:
#     js_result, prec_recall_f = evaluate_vague_final(model_best_from_valid, test_loader, valid_loader, R, device)
# if args.nonVaguePred:
#     acc_nonvague = evaluate_nonvague_final(model_best_from_valid, test_loader, device)
# test_result_log(js_result, prec_recall_f, acc_nonvague, True)

# RAPS
cudnn.benchmark = True
batch_size = 128

# Get your model
_ = model.eval()
model.to(device)

# Get the conformal calibcration dataset

# Initialize loaders 
calib_loader = valid_loader

# Conformalize model
cmodel = ConformalModel(model, valid_loader, alpha=0.1, lamda_criterion='size')

correct_vague = 0.0
correct_nonvague = 0.0
vague_total = 0
nonvague_total = 0

test_ds = mydata.test_loader.dataset

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
    height = 32
    
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
    
    print("preded   set: ", set_predict)
    print("ground truth: ", set_ground)
    print("inter    set: ", inter_set)
    print("union    set: ", union_set)

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
print(pred_singl, label_singl, num_corr_1)
print(pred_singl, set_ground, num_corr_2)
print(pred_singl, set_ground, num_corr_3)

comp_or_not_original = torch.tensor(comp_or_not_original)
comp_or_not_pred = torch.tensor(comp_or_not_pred)
print("### precision, recall, f1, comp_GT_cnt, cmp_pred_cnt:")
print(precision_recall_f_v1(comp_or_not_original, comp_or_not_pred))

stat_result=[correct_nonvague, correct_vague, nonvague_total, vague_total]

avg_js_nonvague = stat_result[0] / (stat_result[2]+1e-10)
avg_js_vague = stat_result[1] / (stat_result[3]+1e-10)
overall_js = (stat_result[0] + stat_result[1])/(stat_result[2] + stat_result[3]+1e-10)

print(f"Count for JS: {stat_result}")
print(f"JS Overall: {overall_js:.4f}, Comp: {avg_js_vague:.4f}, Singleton: {avg_js_nonvague:.4f}")

acc_1 = num_corr_1 / len(test_ds)
acc_2 = num_corr_2 / len(test_ds)
acc_3 = num_corr_3 / len(test_ds)
print(f"NonVaguePredictions: acc 1: {acc_1:.4f}, acc 2: {acc_2:.4f}, acc3: {acc_3:.4f}")
