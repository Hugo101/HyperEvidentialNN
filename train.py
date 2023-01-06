import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time 
import wandb
from helper_functions import one_hot_embedding
from backbones import EfficientNet_pretrain
from evaluate import train_valid_log, evaluate_model 


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
    
    pretrainedModel = None
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

    dataloader = mydata.train_loader 
    dataset_size_train = len(dataloader.dataset)
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        begin_epoch = time.time()
        print("Training...")
        print(f" get last lr:{scheduler.get_last_lr()}") if scheduler else ""
        model.train()  # Set model to training mode
        
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

            loss.backward()
            optimizer.step()

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

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / dataset_size_train
        epoch_acc = running_corrects / dataset_size_train
        epoch_acc = epoch_acc.detach()

        if exp_type in [1, 4, 5, 6]:
            epoch_loss_1 = running_loss_1 / dataset_size_train
            epoch_loss_2 = running_loss_2 / dataset_size_train
        if exp_type in [2, 3, 7]:
            epoch_loss_1 = running_loss_1 / dataset_size_train
            epoch_loss_2 = running_loss_2 / dataset_size_train
            epoch_loss_3 = running_loss_3 / dataset_size_train

        train_valid_log(exp_type, "train", epoch, epoch_acc, epoch_loss, epoch_loss_1, epoch_loss_2, epoch_loss_3)
        time_epoch_train = time.time() - begin_epoch
        print(
        f"Finish the Train in this epoch in {time_epoch_train//60:.0f}m {time_epoch_train%60:.0f}s.")

        # Validation phase
        valid_acc, valid_loss = evaluate_model(
            model,
            mydata,
            num_classes,
            criterion,
            uncertainty=uncertainty,
            kl_reg=kl_reg,
            kl_lam=kl_lam,
            kl_reg_teacher=kl_reg_teacher,
            kl_lam_teacher=kl_lam_teacher,
            forward_kl_teacher=forward_kl_teacher,
            pretrainedModel=pretrainedModel,
            entropy_reg=entropy_reg,
            entropy_lam=entropy_lam,
            ce_lam=ce_lam,
            exp_type=exp_type,
            device=device,
            epoch = epoch,
        )

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            wandb.run.summary["best_valid_acc"] = valid_acc
            wandb.run.summary["best_epoch"] = best_epoch
            print(f"The best epoch: {best_epoch}, acc: {best_acc:.4f}.")
            best_model_wts = copy.deepcopy(model.state_dict()) # deep copy the model

        time_epoch = time.time() - begin_epoch
        print(f"Finish the EPOCH in {time_epoch//60:.0f}m {time_epoch%60:.0f}s.")
        
        time.sleep(0.5)

    time_elapsed = time.time() - since
    print(f"TRAINing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.")

    final_model_wts = copy.deepcopy(model.state_dict()) # view the model in the last epoch is the best 
    model.load_state_dict(final_model_wts)

    print(f"Best val epoch: {best_epoch}, Acc: {best_acc:4f}")
    model_best = copy.deepcopy(model)
    # load best model weights
    model_best.load_state_dict(best_model_wts)

    return model, model_best, best_epoch
