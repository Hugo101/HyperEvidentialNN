import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import time 
import wandb
from helper_functions import one_hot_embedding
from GDD_test import evaluate_vague_nonvague


def train_valid_log(expType, phase, epoch, acc, loss, epoch_loss_1, epoch_loss_2, epoch_loss_3):
    if expType == 0:
        wandb.log({f"{phase}_epoch": epoch, f"{phase}_loss": loss, f"{phase}_acc": acc}, step=epoch)
        print(f"{phase.capitalize()} loss: {loss:.4f} acc: {acc:.4f}")

    if expType == 101:
        wandb.log({
            f"{phase}_epoch": epoch, f"{phase}_loss": loss, 
            f"{phase}_loss_1_uce": epoch_loss_1, 
            f"{phase}_loss_2_entr": epoch_loss_2,
            f"{phase}_loss_3_l2": epoch_loss_3,  
            f"{phase}_acc": acc,  "epoch": epoch})
        print(
            f"{phase.capitalize()} loss: {loss:.4f} \
                (loss_1_uce: {epoch_loss_1:.4f}, \
                    loss_2_entr:{epoch_loss_2:.4f}) \
                        loss_3_l2:{epoch_loss_3:.4f}) \
                    acc: {acc:.4f}")


def evaluate_model(
    model,
    mydata,
    num_classes,
    criterion,
    uncertainty=False,
    entropy_lam=0.001,
    l2_lam=1,
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
            batch_size = inputs.size(0)
            
            # forward
            if uncertainty:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                loss = 0.
                loss_first = 0.
                loss_second = 0.
                loss_third = 0.
                for i in range(batch_size): 
                    #HENN GDD
                    loss_one_example, loss_first_i, loss_second_i, loss_third_i, _ = criterion(
                        outputs[i], labels[i], mydata.R, epoch, mydata.num_classes, 
                        entropy_lam, l2_lam, 
                        device=device)
                    loss += loss_one_example
                    loss_first += loss_first_i
                    loss_second += loss_second_i
                    loss_third += loss_third_i
                loss = loss / batch_size
                loss_first = loss_first / batch_size
                loss_second = loss_second / batch_size
                loss_third = loss_third / batch_size

            else: #cross entropy 
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # statistics
            # batch_size = inputs.size(0)
            running_loss += loss.detach() * batch_size
            running_corrects += torch.sum(preds == labels)
            
            if exp_type in [101]:
                running_loss_1 += loss_first * batch_size
                running_loss_2 += loss_second * batch_size
                running_loss_3 += loss_third * batch_size

    valid_loss = running_loss / dataset_size
    valid_acc = running_corrects / dataset_size
    valid_acc = valid_acc.detach()

    if exp_type in [101]:
        epoch_loss_1 = running_loss_1 / dataset_size
        epoch_loss_2 = running_loss_2 / dataset_size
        epoch_loss_3 = running_loss_3 / dataset_size

    train_valid_log(exp_type, "valid", epoch, valid_acc, valid_loss,epoch_loss_1, epoch_loss_2, epoch_loss_3)
    
    time_epoch = time.time() - begin_eval
    print(
    f"Finish the evaluation in this epoch in {time_epoch//60:.0f}m {time_epoch%60:.0f}s.")

    evaluate_vague_nonvague(
        model, mydata.test_loader, mydata.R, 
        mydata.num_classes, mydata.num_comp, 
        mydata.vague_classes_ids, 
        epoch, device)

    # print("valid_acc", valid_acc, valid_acc_2)
    return valid_acc, valid_loss
