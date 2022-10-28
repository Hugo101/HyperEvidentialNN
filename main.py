# Import libraries
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from random import randint
from datetime import datetime as dt
import time

import json          
from sklearn import metrics
import urllib
import zipfile
import csv
import math
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data import Subset
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision import models, datasets

from config import parser  
import common_tools as ct 
from data.tinyImageNet import tinyImageNetVague 
from backbones import HENN_EfficientNet, HENN_ResNet50, HENN_VGG16
from helper_functions import lossFunc, numAccurate, evaluate_vague_nonvague, draw_roc

args = parser.parse_args("")
# print(args)


@torch.no_grad()
def evaluate(model, val_loader, num_single_classes, kappa, a_copy, 
             annealing_coefficient, device):
    model.eval()
    results = {
                'accuracy': 0.0,
                'mean_val_loss': 0.0
              }
    total_correct = 0.0
    total_samples = 0
    val_losses = []
    for batch in val_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=kappa)
        output = model(images)
        loss = lossFunc(output, one_hot_labels, a_copy, num_single_classes, annealing_coefficient)
        # batch_loss, _, _ = lossFunc(r, one_hot_labels, a_copy, annealing_coefficient)
        # loss = torch.mean(batch_loss)

        total_correct += numAccurate(output, labels)
        total_samples += len(labels)
        val_loss = loss.detach()
        val_losses.append(val_loss)
    results['mean_val_loss'] = torch.stack(val_losses).mean().item()
    results['accuracy'] = total_correct / total_samples
    return results

def train(epochs, model, lr, train_loader, val_loader, 
          num_single_classes, kappa, a_copy, device, 
          weight_decay=0, model_save_dir=""):
#     ignored_params = list(map(id, model.fc.parameters()))
#     base_params = filter(lambda p: id(p) not in ignored_params,
#                          model.parameters())

#     optimizer = torch.optim.Adam([
#                 {'params': base_params},
#                 {'params': model.fc.parameters(), 'lr': lr}
#             ], lr=lr*0.1, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs_stage_1], gamma=0.1)
    global_step = 0
    annealing_step = 1000.0 * args.n_batches
    for epoch in range(epochs):
        # Training Phase 
        print(f" get last lr:{scheduler.get_last_lr()}")
        model.train()
        train_losses = []
        squared_losses = []
        kls = []
        for batch in train_loader:
            optimizer.zero_grad()
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=kappa)
            output = model(images)
            annealing_coefficient = min(1.0, global_step/annealing_step)
            loss = lossFunc(output, one_hot_labels, a_copy, num_single_classes, annealing_coefficient)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            # batch_loss, squared_loss, kl = lossFunc(r, one_hot_labels, a_copy, annealing_coefficient)
            # loss = torch.mean(batch_loss)
            # train_losses.append(loss)
            # squared_losses.append(torch.mean(squared_loss))
            # kls.append(torch.mean(kl))
            # loss.backward()
            # optimizer.step()
            global_step += 1
        mean_train_loss = torch.stack(train_losses).mean().item()
        # mean_squared_loss = torch.stack(squared_losses).mean().item()
        # mean_kl_loss = torch.stack(kls).mean().item()
        
        # Validation phase
        results = evaluate(model, val_loader, annealing_coefficient)
        print(f"Epoch [{epoch}], Mean Training Loss: {mean_train_loss:.4f}, Mean Validation Loss: {results['mean_val_loss']:.4f}, Validation Accuracy: {results['accuracy']:.4f}")
    
        if (epoch + 1) % 5 == 0:
            saved_path = os.path.join(model_save_dir, f'model_traditional_{epoch}.pt')
            torch.save(model.state_dict(), saved_path)
            # torch.save(model.state_dict(), PATH_HENN)
    return model 


def main():
    num_comp = args.num_comp
    save_dir = args.save_dir
    model_save_dir = os.path.join(save_dir, "models_ResNet")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    PATH_HENN = os.path.join(model_save_dir, "model_HENN.pt")

    ct.set_logger('{}/log_file_outerLossAcc_seed_{}'.format(args.output_subfolder, args.seed), 'err')   # log file
    # ct.set_logger('{}/log_file_seed_{}'.format(args.output_subfolder, args.seed), 'out')   # log file
    print('Random Seed: {}'.format(args.seed))
    ct.set_random_seeds(args.seed)
    device = ct.set_device(args.gpu_id)

    if args.dataset == "tinyImageNet":
        dataset = tinyImageNetVague()
        args.nbatches = dataset.nbatches
    # if args.dataset == "cifar100":
        # dataset = cifar100Vague()
    train_dl, valid_dl = dataset.train_dl, dataset.valid_dl
    test_dl = dataset.test_dl

    if args.backbone == "EfficientNet":
        model = HENN_EfficientNet(dataset.kappa).to(device)
    if args.backbone == "ResNet50":
        model = HENN_ResNet50(dataset.kappa).to(device)
    if args.backbone == "VGG16":
        model = HENN_VGG16(dataset.kappa).to(device)
    
    W = dataset.num_classes 
    a = torch.div(torch.ones(dataset.num_classes), dataset.num_classes).to(device)
    vague_classes_ids = dataset.vague_classes_ids
    num_comp = dataset.num_comp

    a_copy = a
    for element in range(num_comp):
        sum_base_rates = torch.zeros(1).to(device)
        for l in vague_classes_ids[element]:
            sum_base_rates += a[l]
        a_copy = torch.cat((a_copy, sum_base_rates.view(1)))

    R = [[el] for el in range(dataset.num_classes)] # actual final labels
    for el in vague_classes_ids:
        R.append(el)

    print(f"a_copy:\n Shape:{a_copy.shape} \n{a_copy}\n R: {R}")

    start = time.time()
    model = train(args.num_epochs, model, args.lr, train_dl, valid_dl, 0)
    end = time.time()
    torch.save(model.state_dict(), PATH_HENN)
    print(f'Total training time for HENN: %s seconds.'%str(end-start))

    #Evaluation, Inference  
    # # todo: add if else for load model 
    # model = to_device(TinyImagenetResNet(), device)
    # model.load_state_dict(torch.load(PATH_HENN))
    # model.eval()

    results = evaluate_vague_nonvague(model, test_dl)
    draw_roc(model, test_dl)

    avg_js_vague = results[0] / results[2]
    print(f"Average Jaccard Similarity Vague Classes:{avg_js_vague}")
    avg_js_nonvague = results[1] / results[3]
    print(f"Average Jaccard Similarity Nonvague Classes:{avg_js_nonvague}")
    overall_acc = (results[0] + results[1])/(results[2] + results[3])
    print(f"Overall Accuracy: {overall_acc}")


    # start = time.time()  # float
    # ct.create_path(args.output_folder)
    # specific_file_name, tag = "", ""
    # # base_path/ssl_path/N-way K-shot/specific_model
    # # base model folder, storing all results of all experiments
    # base_path = os.path.join(args.output_folder, f"{args.dataset}_{args.scenario}")
    # # specific model folder, storing the specific model (specific combination in the configure file)
    
    # few_shot_path = os.path.join(ssl_path, f"#way_{args.num_ways}_#shot_{args.num_shots}")

    # if args.ssl_algo in ["SMI", "SMIcomb"]:
    #     tag = '_'.join(['BudgetS', str(args.budget_s), 'BudgetQ', str(args.budget_q), "TrueLabel", str(args.select_true_label)])
    
    # if not args.resume:
    #     specific_file_name = time.strftime('%Y-%m-%d-%H%M%S') + "_" + tag + "_" + '_'.join(
    #         ['LabelRatio', str(args.ratio), '#ShotU', str(args.num_shots_unlabeled), '#InnerLR', str(args.step_size), 'Seed', str(args.seed), ])
    #     specific_model_path = os.path.join(few_shot_path, specific_file_name)
    #     ct.create_path(specific_model_path)
    #     args.output_subfolder = os.path.abspath(specific_model_path)   # absolute path
    # else:
    #     args.output_subfolder = os.path.abspath(args.checkpoint_path)  # absolute path

    # args.model_path = os.path.abspath(os.path.join(args.output_subfolder, 'best_model'))
    # args.result_path = os.path.abspath(os.path.join(args.output_subfolder, 'best_model_valid_test_result.pkl'))
    # # save the config, json is better here because it is easily to open directly
    # with open(os.path.join(args.output_subfolder, 'config.json'), 'w') as f:
    #     json.dump(vars(args), f, indent=2)


if __name__ == "__main__":



    # results = evaluate_vague_nonvague(model, test_dl)

    # draw_roc(model, test_dl)

    # avg_js_vague = results[0] / results[2]
    # print(f"Average Jaccard Similarity Vague Classes:{avg_js_vague}")

    # avg_js_nonvague = results[1] / results[3]
    # print(f"Average Jaccard Similarity Nonvague Classes:{avg_js_nonvague}")

    # overall_acc = (results[0] + results[1])/(results[2] + results[3])
    # print(f"Overall Accuracy: {overall_acc}")