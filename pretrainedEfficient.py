import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import copy
import time
import yaml
import wandb
import torch
from torch import optim, nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter

from config_args import parser
from common_tools import create_path, set_device, dictToObj, set_random_seeds
from data.tinyImageNet import tinyImageNetVague
from data.cifar100 import CIFAR100Vague
from backbones import EfficientNet_pretrain

args = parser.parse_args()
opt = vars(args)
# build the path to save model and results
create_path(args.output_folder) 
base_path = os.path.join(args.output_folder, args.saved_spec_dir)
create_path(base_path)

config_file = os.path.join(base_path, "config.yml")
config = yaml.load(open(config_file), Loader=yaml.FullLoader)
opt.update(config)

# config = {
#     "dataset": "tinyimagenet",
#     "backbone": "EfficientNet-b3",
#     "epochs": 3,
#     "init_lr": 0.0001,
# }

# convert args from Dict to Object
args = dictToObj(opt)
device = set_device(args.gpu)

def train_log(phase, epoch, acc, loss):
    wandb.log({
        f"{phase} epoch": epoch, 
        f"{phase} loss": loss, 
        f"{phase} acc": acc}, step=epoch)
    print(f"{phase.capitalize()} loss: {loss:.4f} acc: {acc:.4f}")

def test_log(phase, epoch, acc, acc_comps, loss):
    wandb.log({
        f"{phase} epoch": epoch, 
        f"{phase} loss": loss, 
        f"{phase} acc": acc, 
        f"{phase} acc_comps": acc_comps}, step=epoch)
    print(f"{phase.capitalize()} loss: {loss:.4f}, acc: {acc:.4f}, acc_comps: {acc_comps:.4f}")
    
def train_pretrain(
    model,
    mydata,
    num_singles,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device=None,
):
    wandb.watch(model, log="all", log_freq=100)

    since = time.time()
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
                print(f" get last lr:{scheduler.get_last_lr()}") if not scheduler else ""
                model.train()  # Set model to training mode
                dataloader = mydata.train_loader 
            else:
                print("Validating...")
                model.eval()  # Set model to evaluate mode
                dataloader = mydata.valid_loader

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for batch_idx, (inputs, label_singl, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
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

            if scheduler is not None:
                if phase == "train":
                    scheduler.step()

            # print(f"##### length of datasets at phase {phase}: {len(dataloader.dataset)}") #pass 
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / len(dataloader.dataset)
            epoch_acc = epoch_acc.cpu().item()

            train_log(phase, epoch, epoch_acc, epoch_loss)

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
                    acc, acc_comps, loss_t, state = evaluate_pretrain(model, mydata.test_loader, num_singles, criterion, device)
                    torch.save(state, f'{base_path}/tiny_{epoch}_{acc:.4f}.pt')
                    test_log("Test", epoch, acc, acc_comps, loss_t)

        time_epoch = time.time() - begin_epoch
        print(f"Finish the EPOCH in {time_epoch//60:.0f}m {time_epoch%60:.0f}s.")

    time_elapsed = time.time() - since
    print(f"TRAINing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.")
    
    print(f"Best val epoch: {best_epoch}, Acc: {best_acc:4f}")
    final_model_wts = copy.deepcopy(model.state_dict()) # view the model in the last epoch is the best 
    model.load_state_dict(final_model_wts)

    model_best = copy.deepcopy(model)
    # load best model weights
    model_best.load_state_dict(best_model_wts)

    return model, model_best, best_epoch


@torch.no_grad()
def evaluate_pretrain(
    model, 
    val_loader,
    num_singles,
    criterion,
    device,
    ):
    model.eval()
    total_correct = 0.0
    total_samples = 0
    val_losses = []
    labels_true_all = []
    labels_pred_all = []
    for batch in val_loader:
        images, labels_singl, labels = batch
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        _, preds = torch.max(output, 1)
        total_correct += torch.sum(preds == labels.data)
        total_samples += len(labels)
        val_loss = loss.detach()
        val_losses.append(val_loss)
        labels_true_all.append(labels)
        labels_pred_all.append(preds)
    labels_true_all = torch.cat(labels_true_all, dim=0).cpu()
    labels_pred_all = torch.cat(labels_pred_all, dim=0).cpu()
    total_correct_2 = torch.sum(labels_true_all == labels_pred_all).item()
    assert total_correct_2 == total_correct
    assert total_samples == len(labels_true_all)
    comp_idx = labels_true_all > num_singles-1
    labels_true_comps = labels_true_all[comp_idx]
    labels_pred_comps = labels_pred_all[comp_idx]
    corr_comps = torch.sum(labels_true_comps == labels_pred_comps).item()
    acc_comps = corr_comps / len(labels_true_comps)
    
    loss = torch.stack(val_losses).mean().item()
    acc = total_correct / total_samples
    state = {
        "model_state_dict": model.state_dict(),
    }
    return acc, acc_comps, loss, state


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
            duplicate=False)
        
    elif args.dataset == "cifar100":
        mydata = CIFAR100Vague(
            args.data_dir,
            num_comp=args.num_comp,
            batch_size=args.batch_size,
            duplicate=False
        )
    num_singles = mydata.num_classes
    num_comps = mydata.num_comp
    print(f"Data: {args.dataset}, num of singleton and composite classes: {num_singles, num_comps}")
    
    num_classes_both = num_singles + num_comps
    if args.backbone == "EfficientNet-b3":
        model = EfficientNet_pretrain(num_classes_both)
    else:
        print(f"### ERROR: The backbone {args.backbone} is invalid!")
    model = model.to(device)
    print("### Loss type: CrossEntropy (no uncertainty)")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    return mydata, model, num_singles, criterion, optimizer, scheduler


def main():
    print('Random Seed: {}'.format(args.seed))
    set_random_seeds(args.seed)

    mydata, model, num_singles, criterion, optimizer, scheduler = make(args)

    if args.train:
        start = time.time()
        model, model_best, epoch_best = train_pretrain(
                                            model,
                                            mydata,
                                            num_singles,
                                            criterion,
                                            optimizer,
                                            scheduler=scheduler,
                                            num_epochs=args.epochs,
                                            device=device,
                                            )

        state = {
            "epoch_best": epoch_best,
            "model_state_dict": model.state_dict(),
            "model_state_dict_best": model_best.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        saved_path = os.path.join(base_path, "model_CrossEntropy.pt")
        torch.save(state, saved_path)
        print(f"Saved: {saved_path}")
        end = time.time()
        print(f'Total training time for ENN: {(end-start)//60:.0f}m {(end-start)%60:.0f}s')

    if args.test:
        test_loader = mydata.test_loader
        saved_path = os.path.join(base_path, "model_CrossEntropy.pt")
        checkpoint = torch.load(saved_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        model_best_from_valid = copy.deepcopy(model)
        model_best_from_valid.load_state_dict(checkpoint["model_state_dict_best"]) 

        # model after the final epoch
        acc, acc_comps, loss_t, state = evaluate_pretrain(model, test_loader, num_singles, criterion, device)
        test_log("TestF", None, acc, acc_comps, loss_t)
        print(f"model after final epoch: acc:{acc:.4f}, acc_comps:{acc_comps:.4f}")

        print(f"### Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:\n")
        acc, acc_comps, loss_t, state = evaluate_pretrain(model, test_loader, num_singles, criterion, device)
        test_log("TestB", None, acc, acc_comps, loss_t)
        print(f"model from validation: acc:{acc:.4f}, acc_comps:{acc_comps:.4f}")


if __name__ == "__main__":
    # tell wandb to get started
    print(config)
    with wandb.init(project=f"ENN-Vague-{config['dataset']}-Pretrained", config=config):
        config = wandb.config
        main()
