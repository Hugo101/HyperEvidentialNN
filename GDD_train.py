import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import copy
import time 
import wandb

from helper_functions import one_hot_embedding, multi_hot_embedding
from GDD_evaluate import train_valid_log, evaluate_model 


def train_batch_log(iteration, acc, loss, epoch_loss_1, epoch_loss_2, epoch_loss_3):
    phase = "trainBatch"
    wandb.log({
        f"{phase}_iteration": iteration, f"{phase}_loss": loss, 
        f"{phase}_loss_1_uce": epoch_loss_1, 
        f"{phase}_loss_2_entr": epoch_loss_2,
        f"{phase}_loss_3_l2": epoch_loss_3,  
        f"{phase}_acc": acc, "batch": iteration})



def train_model(
    model,
    mydata,
    num_classes,
    criterion,
    optimizer,
    args,
    scheduler=None,
    device=None,
):
    num_epochs=args.epochs
    uncertainty=args.use_uncertainty
    entropy_lam=args.entropy_lam
    l2_lam=args.l2_lam
    exp_type=args.exp_type
    
    wandb.watch(model, log="all", log_freq=100)

    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    dataloader = mydata.train_loader 
    dataset_size_train = len(dataloader.dataset)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
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
            batch_size = inputs.size(0)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            if uncertainty:
                outputs = model(inputs) # evidence output
                _, preds = torch.max(outputs, 1)

                loss = 0.
                loss_first = 0.
                loss_second = 0.
                loss_third = 0.
                for i in range(batch_size): 
                    #HENN GDD
                    loss_one_example, loss_first_i, loss_second_i, loss_third_i, flag_singleton = criterion(
                        outputs[i], labels[i], mydata.R, epoch, mydata.num_classes,  # num of singletons
                        entropy_lam, l2_lam, 
                        device=device)

                    if epoch%10==0 and batch_idx in [66, 67, 68]:
                        print(f"### Epoch {epoch} - batch {batch_idx}/{len(dataloader)}, %%% Example {i}/{batch_size}, Flag_singleton: {flag_singleton}, EntropyCt: {loss_second_i}, EntropyAll: {loss_second}, Evidence: {outputs[i].data.cpu()}")
                    #     if i==142:
                    #         print(f"#### Example {i}/{batch_size}, EntropyCt: {loss_second_i}, EntropyAll: {loss_second}")

                    loss += loss_one_example
                    loss_first += loss_first_i
                    loss_second += loss_second_i
                    loss_third += loss_third_i
                    
                    # print(f"##Ep: {epoch}- batch {batch_idx}/{len(dataloader)}. ExampleID {i}/{batch_size}, CurrentEntropy: {loss_second_i:.4f}, CurrSumEntropy: {loss_second:.4f}")
                    
                loss = loss / batch_size
                loss_first = loss_first / batch_size
                loss_second = loss_second / batch_size
                loss_third = loss_third / batch_size
                
            else: #cross entropy 
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            acc_batch = torch.sum(preds == labels)/batch_size
            iteration = epoch * len(dataloader) + batch_idx
            train_batch_log(iteration, acc_batch, loss, loss_first, loss_second, loss_third)
            print(f"##Epoch {epoch} - batch {batch_idx}/{len(dataloader)} loss: {loss:.4f}, loss_first: {loss_first:.4f}, loss_second: {loss_second:.4f}, loss_third: {loss_third:.4f}, acc: {acc_batch:.4f}")
            # print(f"output: {outputs[0]}")
            # if batch_idx == 67:
            #     print(f"## batch {batch_idx}")
            #     print(f"output: {outputs}")
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            # batch_size = inputs.size(0)
            running_loss += loss.detach() * batch_size
            running_corrects += torch.sum(preds == labels)
            
            running_loss_1 += loss_first * batch_size
            running_loss_2 += loss_second * batch_size
            running_loss_3 += loss_third * batch_size

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / dataset_size_train
        epoch_acc = running_corrects / dataset_size_train
        epoch_acc = epoch_acc.detach()

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
            entropy_lam=entropy_lam,
            l2_lam=l2_lam,
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