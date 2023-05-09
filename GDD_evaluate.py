import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import time 
import wandb
from helper_functions import one_hot_embedding
from GDD_test import evaluate_vague_nonvague


def train_valid_log(expType, phase, epoch, acc, loss, epoch_loss_1, epoch_loss_2, epoch_loss_3, epoch_loss_4):
    if expType == 0:
        wandb.log({f"{phase}_epoch": epoch, f"{phase}_loss": loss, f"{phase}_acc": acc}, step=epoch)
        print(f"{phase.capitalize()} loss: {loss:.4f} acc: {acc:.4f}")

    if expType == 101:
        wandb.log({
            f"{phase}_epoch": epoch, f"{phase}_loss": loss, 
            f"{phase}_loss_1_uce": epoch_loss_1, 
            f"{phase}_loss_2_entrDir": epoch_loss_2,
            f"{phase}_loss_3_entrGDD": epoch_loss_3,
            f"{phase}_loss_4_kl": epoch_loss_4,
            f"{phase}_acc": acc,  "epoch": epoch})
        print(
            f"{phase.capitalize()} loss: {loss:.4f} \
                (loss_1_uce: {epoch_loss_1:.4f}, \
                 loss_2_entrDir:{epoch_loss_2:.4f}) \
                 loss_3_entrGDD:{epoch_loss_3:.4f}) \
                 loss_4_kl:{epoch_loss_4:.4f}) \
                 acc: {acc:.4f}")


def evaluate_model(
    model,
    mydata,
    criterion,
    args,
    device=None,
    epoch = 1,
):
    uncertainty=args.use_uncertainty
    entropy_lam_Dir=args.entropy_lam_Dir
    entropy_lam_GDD=args.entropy_lam_GDD
    kl_lam = args.kl_lam
    l2_lam=args.l2_lam
    kl_reg=args.kl_reg
    exp_type=args.exp_type
    kl_anneal = args.kl_anneal
    
    begin_eval = time.time()

    print("Validing...")
    model.eval()  # Set model to eval mode
    dataloader = mydata.valid_loader 
    dataset_size = len(dataloader.dataset)

    running_loss = 0.0
    running_loss_1, running_loss_2, running_loss_3 = 0.0, 0.0, 0.0
    epoch_loss_1,    epoch_loss_2,    epoch_loss_3 = 0.0, 0.0, 0.0
    running_corrects = 0.0

    running_loss_4 = 0.0
    epoch_loss_4 = 0.0
    singleton_size = 0
    # Iterate over data.
    with torch.no_grad():
        for batch_idx, (inputs, targets_GT, labels) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            targets_GT = targets_GT.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            
            singleton_size += torch.sum(labels < mydata.num_classes) # num of singletons

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # loss = 0.
            # loss_first = 0.
            # loss_second = 0.
            # loss_third = 0.
            # loss_fourth = 0.
            # singleton_size = 0
            # for i in range(batch_size): 
            #     #HENN GDD
            #     loss_one_example, loss_first_i, loss_second_i, loss_third_i, loss_fourth_i, flag_singleton = criterion(
            #         outputs[i], labels[i], mydata.R, epoch, mydata.num_classes, 
            #         entropy_lam, l2_lam, entropy_lam_Dir,
            #         device=device)

            #     singleton_size += flag_singleton
            #     loss += loss_one_example
            #     loss_first += loss_first_i
            #     loss_second += loss_second_i
            #     loss_third += loss_third_i
            #     loss_fourth += loss_fourth_i
            
            loss, loss_first_avg, loss_second_avg, loss_third_avg, loss_fourth_avg = criterion(
                                                    outputs, 
                                                    labels, 
                                                    mydata.R,
                                                    epoch, 
                                                    mydata.num_classes,
                                                    args.anneal_step, 
                                                    kl_lam,
                                                    entropy_lam_Dir,
                                                    entropy_lam_GDD,
                                                    anneal=kl_anneal,
                                                    kl_reg=kl_reg,
                                                    device=device)
            # loss_third_avg = 0
            # loss_fourth_avg = 0 

            # statistics
            running_loss += loss.detach()
            running_corrects += torch.sum(preds == labels)
            
            running_loss_1 += loss_first_avg * batch_size
            running_loss_2 += loss_second_avg * batch_size
            running_loss_3 += loss_third_avg * batch_size
            running_loss_4 += loss_fourth_avg * batch_size

    valid_loss = running_loss / dataset_size
    valid_acc = running_corrects / dataset_size
    valid_acc = valid_acc.detach()

    epoch_loss_1 = running_loss_1 / dataset_size
    epoch_loss_2 = running_loss_2 / dataset_size
    epoch_loss_3 = running_loss_3 / dataset_size
    epoch_loss_4 = running_loss_4 / dataset_size

    train_valid_log(exp_type, "valid", epoch, valid_acc, valid_loss, epoch_loss_1, epoch_loss_2, epoch_loss_3, epoch_loss_4)
    
    composite_size = dataset_size - singleton_size
    print(f"singletonSize:CompositeSize = {singleton_size}:{composite_size}")
    
    time_epoch = time.time() - begin_eval
    print(
    f"Finish the evaluation in this epoch in {time_epoch//60:.0f}m {time_epoch%60:.0f}s.")

    valid_acc_GT, valid_overJS = evaluate_vague_nonvague(
        model, mydata.valid_loader, mydata.R, 
        mydata.num_classes, mydata.num_comp, 
        mydata.vague_classes_ids, 
        epoch, device, train_flag=2)

    evaluate_vague_nonvague(
        model, mydata.test_loader, mydata.R, 
        mydata.num_classes, mydata.num_comp, 
        mydata.vague_classes_ids, 
        epoch, device, train_flag=3)

    # print("valid_acc", valid_acc, valid_acc_2)
    return valid_acc, valid_loss, valid_acc_GT, valid_overJS
