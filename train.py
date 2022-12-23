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
# saved_path_pretrain = "/data/cxl173430/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/models_pretrained/tiny_2_0.83.pkl"

#PDML2:
# saved_path_pretrain = "/home/cxl173430/Documents/projects/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/models_pretrained/tiny_2_0.83.pkl"

# saved_path_pretrain = "/mnt/data/home/cxl173430/Documents/projects/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results/tiny/pretrain_lre-5/tiny_29_0.8490.pt"

#PDML4:
# saved_path_pretrain = "/home/cxl173430/Projects/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/models_pretrained/tiny_2_0.83.pkl"

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
            f"{phase} loss_3_kl_pretrain": epoch_loss_3,  
            f"{phase} acc": acc}, step=epoch)
        print(
            f"{phase.capitalize()} loss: {loss:.4f} \
                (loss_1: {epoch_loss_1:.4f}, \
                    loss_2_kl:{epoch_loss_2:.4f}) \
                        loss_3_kl_pretrain:{epoch_loss_3:.4f}) \
                    acc: {acc:.4f}")
    if expType == 4:
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
    kl_reg_pretrain=False,
    kl_lam_pretrain=0.001,
    entropy_reg=False,
    entropy_lam=0.001,
    forward_kl_pretrain=True,
    exp_type=0,
    saved_path_pretrain=None,
    device=None,
    logdir="./runs"
):
    wandb.watch(model, log="all", log_freq=100)

    since = time.time()
    
    if exp_type == 3:
        pretrainedModel = EfficientNet_pretrain(num_classes)
        checkpoint = torch.load(saved_path_pretrain, map_location=device)
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

            running_loss = 0.0
            running_loss_1, running_loss_2, running_loss_3 = 0.0, 0.0, 0.0
            epoch_loss_1,    epoch_loss_2,    epoch_loss_3 = 0.0, 0.0, 0.0
            running_corrects = 0.0

            # Iterate over data.
            for batch_idx, (inputs, _, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    if uncertainty:
                        y = one_hot_embedding(labels, num_classes, device)
                        y = y.to(device)
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
                                None, kl_lam, None, None, None, None,
                                kl_reg=kl_reg,
                                exp_type=exp_type, 
                                device=device)
                        if exp_type == 3: #expected_CE + KL + KL_pretrain
                            with torch.no_grad():
                                logits = pretrainedModel(inputs)
                                pretrainedProb = F.softmax(logits, dim=1)
                            loss, loss_first, loss_second, loss_third = criterion(
                                outputs, y.float(), epoch, num_classes, 
                                None, kl_lam, kl_lam_pretrain, None, 
                                pretrainedProb, forward_kl_pretrain,
                                kl_reg=kl_reg, kl_reg_pretrain=kl_reg_pretrain,
                                exp_type=exp_type,
                                device=device)
                            pass 
                        if exp_type == 4: #expected_CE + Entropy
                            loss, loss_first, loss_second = criterion(
                                outputs, y.float(), epoch, num_classes, 
                                None, 0, None, entropy_lam, None, None,
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
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                
                if exp_type in [1, 4]:
                    running_loss_1 += loss_first * batch_size
                    running_loss_2 += loss_second * batch_size
                if exp_type in [2, 3]:
                    running_loss_1 += loss_first * batch_size
                    running_loss_2 += loss_second * batch_size
                    running_loss_3 += loss_third * batch_size

            if scheduler is not None:
                if phase == "train":
                    scheduler.step()

            # print(f"##### length of datasets at phase {phase}: {len(dataloader.dataset)}") #pass 
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            epoch_acc = epoch_acc.detach().cpu().item()

            if exp_type in [1, 4]:
                epoch_loss_1 = running_loss_1 / len(dataloader.dataset)
                epoch_loss_2 = running_loss_2 / len(dataloader.dataset)
            if exp_type in [2, 3]:
                epoch_loss_1 = running_loss_1 / len(dataloader.dataset)
                epoch_loss_2 = running_loss_2 / len(dataloader.dataset)
                epoch_loss_3 = running_loss_3 / len(dataloader.dataset)

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
                    state = {
                        "model_state_dict": model.state_dict(),
                    }
                    torch.save(state, f'{logdir}/HENN_{epoch}_{acc:.4f}.pt')

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



# def train_HENN(
#     model,
#     mydata,
#     epochs, 
#     lr,
#     num_single_classes, kappa, a_copy, device, 
#     weight_decay=0, model_save_dir=""):
# #     ignored_params = list(map(id, model.fc.parameters()))
# #     base_params = filter(lambda p: id(p) not in ignored_params,
# #                          model.parameters())

# #     optimizer = torch.optim.Adam([
# #                 {'params': base_params},
# #                 {'params': model.fc.parameters(), 'lr': lr}
# #             ], lr=lr*0.1, weight_decay=weight_decay)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs_stage_1], gamma=0.1)
#     global_step = 0
#     # annealing_step = 1000.0 * args.n_batches
#     for epoch in range(epochs):
#         # Training Phase 
#         print(f" get last lr:{scheduler.get_last_lr()}")
#         model.train()
#         train_losses = []
#         squared_losses = []
#         kls = []
#         for batch in train_loader:
#             optimizer.zero_grad()
#             images, labels = batch
#             images, labels = images.to(device), labels.to(device)
#             # one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=kappa)
#             output = model(images)
#             # annealing_coefficient = min(1.0, global_step/annealing_step)
#             loss = lossFunc(output, one_hot_labels, a_copy, num_single_classes, annealing_coefficient)
#             train_losses.append(loss)
#             loss.backward()
#             optimizer.step()
#             # batch_loss, squared_loss, kl = lossFunc(r, one_hot_labels, a_copy, annealing_coefficient)
#             # loss = torch.mean(batch_loss)
#             # train_losses.append(loss)
#             # squared_losses.append(torch.mean(squared_loss))
#             # kls.append(torch.mean(kl))
#             # loss.backward()
#             # optimizer.step()
#             global_step += 1
#         mean_train_loss = torch.stack(train_losses).mean().item()
#         # mean_squared_loss = torch.stack(squared_losses).mean().item()
#         # mean_kl_loss = torch.stack(kls).mean().item()
        
#         # Validation phase
#         results = evaluate(model, val_loader, annealing_coefficient)
#         print(f"Epoch [{epoch}], Mean Training Loss: {mean_train_loss:.4f}, Mean Validation Loss: {results['mean_val_loss']:.4f}, Validation Accuracy: {results['accuracy']:.4f}")
    
#         if (epoch + 1) % 5 == 0:
#             saved_path = os.path.join(model_save_dir, f'model_traditional_{epoch}.pt')
#             torch.save(model.state_dict(), saved_path)
#             # torch.save(model.state_dict(), PATH_HENN)
#     return model 
