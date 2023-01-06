import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time 
import wandb
from helper_functions import one_hot_embedding
from test import evaluate_vague_nonvague
from backbones import EfficientNet_pretrain


def train_valid_log(expType, phase, epoch, acc, loss, epoch_loss_1, epoch_loss_2, epoch_loss_3):
    if expType == 0:
        wandb.log({f"{phase}_epoch": epoch, f"{phase}_loss": loss, f"{phase}_acc": acc}, step=epoch)
        print(f"{phase.capitalize()} loss: {loss:.4f} acc: {acc:.4f}")
    if expType == 1:
        wandb.log({
            f"{phase}_epoch": epoch, f"{phase}_loss": loss, 
            f"{phase}_loss_1": epoch_loss_1, 
            f"{phase}_loss_2_kl": epoch_loss_2, 
            f"{phase}_acc": acc}, step=epoch)
        print(
            f"{phase.capitalize()} loss: {loss:.4f}\
                (loss_1: {epoch_loss_1:.4f},\
                    loss_2_kl:{epoch_loss_2:.4f})\
                    acc: {acc:.4f}")
    if expType == 2:
        wandb.log({
            f"{phase}_epoch": epoch, f"{phase}_loss": loss, 
            f"{phase}_loss_1": epoch_loss_1, 
            f"{phase}_loss_2_kl": epoch_loss_2,
            f"{phase}_loss_3_ce": epoch_loss_3,  
            f"{phase}_acc": acc}, step=epoch)
        print(
            f"{phase.capitalize()} loss: {loss:.4f} \
                (loss_1: {epoch_loss_1:.4f}, \
                    loss_2_kl:{epoch_loss_2:.4f}) \
                        loss_3_ce:{epoch_loss_3:.4f}) \
                    acc: {acc:.4f}")
    if expType == 3:
        wandb.log({
            f"{phase}_epoch": epoch, f"{phase}_loss": loss, 
            f"{phase}_loss_1": epoch_loss_1, 
            f"{phase}_loss_2_kl": epoch_loss_2,
            f"{phase}_loss_3_kl_teacher": epoch_loss_3,  
            f"{phase}_acc": acc}, step=epoch)
        print(
            f"{phase.capitalize()} loss: {loss:.4f} \
                (loss_1: {epoch_loss_1:.4f}, \
                    loss_2_kl:{epoch_loss_2:.4f}) \
                        loss_3_kl_teacher:{epoch_loss_3:.4f}) \
                    acc: {acc:.4f}")
    if expType in [4, 5, 6]:
        wandb.log({
            f"{phase}_epoch": epoch, f"{phase}_loss": loss, 
            f"{phase}_loss_1": epoch_loss_1, 
            f"{phase}_loss_2_entropy": epoch_loss_2, 
            f"{phase}_acc": acc}, step=epoch)
        print(
            f"{phase.capitalize()} loss: {loss:.4f} \
                (loss_1: {epoch_loss_1:.4f}, \
                    loss_2_entropy:{epoch_loss_2:.4f}) \
                    acc: {acc:.4f}")

    if expType == 7:
        wandb.log({
            f"{phase}_epoch": epoch, f"{phase}_loss": loss, 
            f"{phase}_loss_1": epoch_loss_1, 
            f"{phase}_loss_2_ce": epoch_loss_2,
            f"{phase}_loss_3_entropy": epoch_loss_3,  
            f"{phase}_acc": acc}, step=epoch)
        print(
            f"{phase.capitalize()} loss: {loss:.4f} \
                (loss_1: {epoch_loss_1:.4f}, \
                    loss_2_ce:{epoch_loss_2:.4f}) \
                        loss_3_entropy:{epoch_loss_3:.4f}) \
                    acc: {acc:.4f}")


def evaluate_model(
    model,
    mydata,
    num_classes,
    criterion,
    uncertainty=False,
    kl_reg=True,
    kl_lam=0.001,
    kl_reg_teacher=False,
    kl_lam_teacher=0.001,
    forward_kl_teacher=True,
    pretrainedModel=None,
    entropy_reg=False,
    entropy_lam=0.001,
    ce_lam=1,
    exp_type=0,
    device=None,
    epoch = 1,
):
    begin_eval = time.time()

    print("Validing...")
    model.eval()  # Set model to eval mode
    dataloader = mydata.valid_loader 
    dataset_size = len(dataloader.dataset)

    running_loss = 0.0
    running_loss_1, running_loss_2, running_loss_3 = 0.0, 0.0, 0.0
    epoch_loss_1,    epoch_loss_2,    epoch_loss_3 = 0.0, 0.0, 0.0
    running_corrects = 0.0

    # Iterate over data.
    with torch.no_grad():
        for batch_idx, (inputs, _, labels) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # forward
            if uncertainty:
                y = one_hot_embedding(labels, num_classes, device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                if exp_type == 1: #expected_MSE + KL
                    loss, loss_first, loss_second = criterion(
                        outputs, y, epoch, num_classes, 
                        None, kl_lam, None, None, 
                        kl_reg=kl_reg, 
                        device=device)
                if exp_type == 2: #expected_CE + KL + CE
                    loss, loss_first, loss_second, loss_third = criterion(
                        outputs, y, epoch, num_classes, 
                        None, kl_lam, None, None, ce_lam, None, None,
                        kl_reg=kl_reg,
                        exp_type=exp_type, 
                        device=device)
                if exp_type == 3: #expected_CE + KL + KL_teacher
                    with torch.no_grad():
                        logits = pretrainedModel(inputs)
                        pretrainedProb = F.softmax(logits, dim=1)
                    loss, loss_first, loss_second, loss_third = criterion(
                        outputs, y, epoch, num_classes, 
                        None, kl_lam, kl_lam_teacher, None, None,
                        pretrainedProb, forward_kl_teacher,
                        kl_reg=kl_reg, kl_reg_teacher=kl_reg_teacher,
                        exp_type=exp_type,
                        device=device)
                if exp_type in [4,5]: #expected_CE - Entropy
                    loss, loss_first, loss_second = criterion(
                        outputs, y, epoch, num_classes, 
                        None, 0, None, entropy_lam, ce_lam, None, None,
                        kl_reg=kl_reg, entropy_reg=entropy_reg,
                        exp_type=exp_type,
                        device=device)
                if exp_type == 6: # CE
                    loss, loss_first, loss_second = criterion(
                        outputs, y, epoch, num_classes, 
                        None, 0, None, entropy_lam, ce_lam, None, None,
                        kl_reg=kl_reg, entropy_reg=entropy_reg,
                        exp_type=exp_type,
                        device=device)
                if exp_type == 7: #expected_CE + CE - Entropy
                    loss, loss_first, loss_second, loss_third = criterion(
                        outputs, y, epoch, num_classes, 
                        None, 0, None, entropy_lam, ce_lam, None, None,
                        kl_reg=kl_reg, entropy_reg=entropy_reg,
                        exp_type=exp_type,
                        device=device)

            else: #cross entropy 
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # statistics
            batch_size = inputs.size(0)
            running_loss += loss.detach() * batch_size
            running_corrects += torch.sum(preds == labels)
            
            if exp_type in [1, 4, 5, 6]:
                running_loss_1 += loss_first * batch_size
                running_loss_2 += loss_second * batch_size
            if exp_type in [2, 3, 7]:
                running_loss_1 += loss_first * batch_size
                running_loss_2 += loss_second * batch_size
                running_loss_3 += loss_third * batch_size

    valid_loss = running_loss / dataset_size
    valid_acc = running_corrects / dataset_size
    valid_acc = valid_acc.detach()

    if exp_type in [1, 4, 5, 6]:
        epoch_loss_1 = running_loss_1 / dataset_size
        epoch_loss_2 = running_loss_2 / dataset_size
    if exp_type in [2, 3, 7]:
        epoch_loss_1 = running_loss_1 / dataset_size
        epoch_loss_2 = running_loss_2 / dataset_size
        epoch_loss_3 = running_loss_3 / dataset_size

    train_valid_log(exp_type, "valid", epoch, valid_acc, valid_loss,epoch_loss_1, epoch_loss_2, epoch_loss_3)
    time_epoch = time.time() - begin_eval
    print(
    f"Finish the evaluation in this epoch in {time_epoch//60:.0f}m {time_epoch%60:.0f}s.")

    valid_acc_2 = evaluate_vague_nonvague(
        model, dataloader, mydata.R, 
        mydata.num_classes, mydata.num_comp, 
        mydata.vague_classes_ids, 
        epoch, device)

    return valid_acc, valid_loss
