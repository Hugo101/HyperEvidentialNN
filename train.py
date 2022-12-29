import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time 
import wandb
from helper_functions import one_hot_embedding
from test import evaluate_vague_nonvague_ENN
from backbones import EfficientNet_pretrain

#PDML:
# saved_path_teacher = "/data/cxl173430/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/models_pretrained/tiny_2_0.83.pkl"

#PDML2:
# saved_path_teacher = "/home/cxl173430/Documents/projects/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/models_pretrained/tiny_2_0.83.pkl"

# saved_path_teacher = "/mnt/data/home/cxl173430/Documents/projects/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results/tiny/pretrain_lre-5/tiny_29_0.8490.pt"

#PDML4:
# saved_path_teacher = "/home/cxl173430/Projects/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/models_pretrained/tiny_2_0.83.pkl"

def train_log(expType, phase, epoch, acc, loss, epoch_loss_1, epoch_loss_2, epoch_loss_3):
    if expType == 0:
        wandb.log({f"{phase} epoch": epoch, f"{phase} loss": loss, f"{phase} acc": acc}, step=epoch)
        print(f"{phase.capitalize()} loss: {loss:.4f} acc: {acc:.4f}")
    if expType == 1:
        wandb.log({
            f"{phase} epoch": epoch, f"{phase} loss": loss, 
            f"{phase} loss_1": epoch_loss_1, 
            f"{phase} loss_2_kl": epoch_loss_2, 
            f"{phase} acc": acc}, step=epoch)
        print(
            f"{phase.capitalize()} loss: {loss:.4f}\
                (loss_1: {epoch_loss_1:.4f},\
                    loss_2_kl:{epoch_loss_2:.4f})\
                    acc: {acc:.4f}")
    if expType == 2:
        wandb.log({
            f"{phase} epoch": epoch, f"{phase} loss": loss, 
            f"{phase} loss_1": epoch_loss_1, 
            f"{phase} loss_2_kl": epoch_loss_2,
            f"{phase} loss_3_ce": epoch_loss_3,  
            f"{phase} acc": acc}, step=epoch)
        print(
            f"{phase.capitalize()} loss: {loss:.4f} \
                (loss_1: {epoch_loss_1:.4f}, \
                    loss_2_kl:{epoch_loss_2:.4f}) \
                        loss_3_ce:{epoch_loss_3:.4f}) \
                    acc: {acc:.4f}")
    if expType == 3:
        wandb.log({
            f"{phase} epoch": epoch, f"{phase} loss": loss, 
            f"{phase} loss_1": epoch_loss_1, 
            f"{phase} loss_2_kl": epoch_loss_2,
            f"{phase} loss_3_kl_teacher": epoch_loss_3,  
            f"{phase} acc": acc}, step=epoch)
        print(
            f"{phase.capitalize()} loss: {loss:.4f} \
                (loss_1: {epoch_loss_1:.4f}, \
                    loss_2_kl:{epoch_loss_2:.4f}) \
                        loss_3_kl_teacher:{epoch_loss_3:.4f}) \
                    acc: {acc:.4f}")
    if expType in [4, 5, 6]:
        wandb.log({
            f"{phase} epoch": epoch, f"{phase} loss": loss, 
            f"{phase} loss_1": epoch_loss_1, 
            f"{phase} loss_2_entropy": epoch_loss_2, 
            f"{phase} acc": acc}, step=epoch)
        print(
            f"{phase.capitalize()} loss: {loss:.4f} \
                (loss_1: {epoch_loss_1:.4f}, \
                    loss_2_entropy:{epoch_loss_2:.4f}) \
                    acc: {acc:.4f}")

    if expType == 7:
        wandb.log({
            f"{phase} epoch": epoch, f"{phase} loss": loss, 
            f"{phase} loss_1": epoch_loss_1, 
            f"{phase} loss_2_ce": epoch_loss_2,
            f"{phase} loss_3_entropy": epoch_loss_3,  
            f"{phase} acc": acc}, step=epoch)
        print(
            f"{phase.capitalize()} loss: {loss:.4f} \
                (loss_1: {epoch_loss_1:.4f}, \
                    loss_2_ce:{epoch_loss_2:.4f}) \
                        loss_3_entropy:{epoch_loss_3:.4f}) \
                    acc: {acc:.4f}")


def train_model(
    model,
    mydata,
    num_classes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    uncertainty=False,
    kl_reg=True,
    kl_lam=0.001,
    kl_reg_teacher=False,
    kl_lam_teacher=0.001,
    forward_kl_teacher=True,
    saved_path_teacher=None,
    entropy_reg=False,
    entropy_lam=0.001,
    ce_lam=1,
    exp_type=0,
    device=None,
    logdir="./runs"
):
    wandb.watch(model, log="all", log_freq=100)

    since = time.time()
    
    if exp_type == 3:
        pretrainedModel = EfficientNet_pretrain(num_classes)
        checkpoint = torch.load(saved_path_teacher, map_location=device)
        # pretrainedModel.load_state_dict(checkpoint["model_state_dict"])
        pretrainedModel.load_state_dict(checkpoint["model_state_dict_best"])
        pretrainedModel.eval()
        pretrainedModel = pretrainedModel.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        begin_epoch = time.time()
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                print("Training...")
                print(f" get last lr:{scheduler.get_last_lr()}") if scheduler else ""
                model.train()  # Set model to training mode
                dataloader = mydata.train_loader 
            else:
                print("Validating...")
                model.eval()  # Set model to evaluate mode
                dataloader = mydata.valid_loader

            data_loader_size = len(dataloader.dataset)
            running_loss = 0.0
            running_loss_1, running_loss_2, running_loss_3 = 0.0, 0.0, 0.0
            epoch_loss_1,    epoch_loss_2,    epoch_loss_3 = 0.0, 0.0, 0.0
            running_corrects = 0.0

            # Iterate over data.
            for batch_idx, (inputs, _, labels) in enumerate(dataloader):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    if uncertainty:
                        y = one_hot_embedding(labels, num_classes, device)
                        y = y.to(device, non_blocking=True)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        if exp_type == 1: #expected_MSE + KL
                            loss, loss_first, loss_second = criterion(
                                outputs, y.float(), epoch, num_classes, 
                                None, kl_lam, None, None, 
                                kl_reg=kl_reg, 
                                device=device)
                        if exp_type == 2: #expected_CE + KL + CE
                            loss, loss_first, loss_second, loss_third = criterion(
                                outputs, y.float(), epoch, num_classes, 
                                None, kl_lam, None, None, ce_lam, None, None,
                                kl_reg=kl_reg,
                                exp_type=exp_type, 
                                device=device)
                        if exp_type == 3: #expected_CE + KL + KL_teacher
                            with torch.no_grad():
                                logits = pretrainedModel(inputs)
                                pretrainedProb = F.softmax(logits, dim=1)
                            loss, loss_first, loss_second, loss_third = criterion(
                                outputs, y.float(), epoch, num_classes, 
                                None, kl_lam, kl_lam_teacher, None, None,
                                pretrainedProb, forward_kl_teacher,
                                kl_reg=kl_reg, kl_reg_teacher=kl_reg_teacher,
                                exp_type=exp_type,
                                device=device)
                        if exp_type in [4,5]: #expected_CE - Entropy
                            loss, loss_first, loss_second = criterion(
                                outputs, y.float(), epoch, num_classes, 
                                None, 0, None, entropy_lam, ce_lam, None, None,
                                kl_reg=kl_reg, entropy_reg=entropy_reg,
                                exp_type=exp_type,
                                device=device)
                        if exp_type == 6: # CE
                            loss, loss_first, loss_second = criterion(
                                outputs, y.float(), epoch, num_classes, 
                                None, 0, None, entropy_lam, ce_lam, None, None,
                                kl_reg=kl_reg, entropy_reg=entropy_reg,
                                exp_type=exp_type,
                                device=device)
                        if exp_type == 7: #expected_CE + CE - Entropy
                            loss, loss_first, loss_second, loss_third = criterion(
                                outputs, y.float(), epoch, num_classes, 
                                None, 0, None, entropy_lam, ce_lam, None, None,
                                kl_reg=kl_reg, entropy_reg=entropy_reg,
                                exp_type=exp_type,
                                device=device)

                    else: #cross entropy 
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                batch_size = inputs.size(0)
                running_loss += loss.detach() * batch_size #todo
                running_corrects += torch.sum(preds == labels)
                
                if exp_type in [1, 4, 5, 6]:
                    running_loss_1 += loss_first * batch_size
                    running_loss_2 += loss_second * batch_size
                if exp_type in [2, 3, 7]:
                    running_loss_1 += loss_first * batch_size
                    running_loss_2 += loss_second * batch_size
                    running_loss_3 += loss_third * batch_size

            if scheduler is not None:
                if phase == "train":
                    scheduler.step()

            # print(f"##### length of datasets at phase {phase}: {data_loader_size}") #pass 
            epoch_loss = running_loss / data_loader_size
            epoch_acc = running_corrects / data_loader_size #todo
            epoch_acc = epoch_acc.detach()

            if exp_type in [1, 4, 5, 6]:
                epoch_loss_1 = running_loss_1 / data_loader_size
                epoch_loss_2 = running_loss_2 / data_loader_size
            if exp_type in [2, 3, 7]:
                epoch_loss_1 = running_loss_1 / data_loader_size
                epoch_loss_2 = running_loss_2 / data_loader_size
                epoch_loss_3 = running_loss_3 / data_loader_size

            train_log(exp_type, phase, epoch, epoch_acc, epoch_loss, epoch_loss_1, epoch_loss_2, epoch_loss_3)
            
            if phase == "train":
                time_epoch_train = time.time() - begin_epoch
                print(
                f"Finish the Train in this epoch in {time_epoch_train//60:.0f}m {time_epoch_train%60:.0f}s.")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                print(f"The best epoch: {best_epoch}, acc: {best_acc}")
                best_model_wts = copy.deepcopy(model.state_dict()) # deep copy the model
            if phase == "val":
                if epoch == 0 or ((epoch+1) % 1 ==0):
                    acc = evaluate_vague_nonvague_ENN(
                        model, mydata.test_loader, mydata.R, 
                        mydata.num_classes, mydata.num_comp, 
                        mydata.vague_classes_ids, 
                        epoch, device)
                    # state = {
                    #     "model_state_dict": model.state_dict(),
                    # }
                    # torch.save(state, f'{logdir}/HENN_{epoch}_{acc:.4f}.pt')

        time_epoch = time.time() - begin_epoch
        print(f"Finish the EPOCH in {time_epoch//60:.0f}m {time_epoch%60:.0f}s.")

    time_elapsed = time.time() - since
    print(f"TRAINing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.")
    
    final_model_wts = copy.deepcopy(model.state_dict()) # view the model in the last epoch is the best 
    model.load_state_dict(final_model_wts)

    print(f"Best val epoch: {best_epoch}, Acc: {best_acc:4f}")
    model_best = copy.deepcopy(model)
    # load best model weights
    model_best.load_state_dict(best_model_wts)

    return model, model_best, best_epoch
