import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import copy
import time
import yaml
import wandb
import torch
from torch import optim, nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter

from config_args import parser
from common_tools import create_path, set_device, dictToObj, set_random_seeds
from data.tinyImageNet import tinyImageNetVague
from data.cifar100 import CIFAR100Vague
from backbones import HENN_EfficientNet, EfficientNet_pretrain
from helper_functions import one_hot_embedding
from loss import edl_mse_loss, edl_digamma_loss, edl_log_loss
from baseline_DetNN import evaluate_nonvague_final
from baseline_DetNN import evaluate_vague_final
from baseline_DetNN import test_result_log
from helper_functions import js_subset

args = parser.parse_args()
opt = vars(args)
# build the path to save model and results
create_path(args.output_folder) 
base_path = os.path.join(args.output_folder, args.saved_spec_dir)
create_path(base_path)

config_file = os.path.join(base_path, "config.yml")
config = yaml.load(open(config_file), Loader=yaml.FullLoader)
opt.update(config)
args = opt

# config = {
#     "dataset": "tinyimagenet",
#     "backbone": "EfficientNet-b3",
#     "uncertainty": True,
#     "epochs": 1,
#     "init_lr": 0.0001,
#     "train": False,
#     "test": True,
#     "nonVaguePred": True,
#     "vaguePred": True,
# }

# convert args from Dict to Object
args = dictToObj(args)
device = set_device(args.gpu)

def train_log(phase, epoch, accDup, accGT, loss):
    wandb.log({
        f"{phase} epoch": epoch, 
        f"{phase} loss": loss, 
        f"{phase} accDup": accDup, 
        f"{phase} accGT": accGT}, step=epoch)
    print(f"{phase.capitalize()} loss: {loss:.4f} accDup: {accDup:.4f} accGT: {accGT:.4f}")


def validate(model, dataloader, criterion, K, epoch):
    print("Validating...")
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects_GT = 0.0
    for batch_idx, (inputs, single_labels_GT, _) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = single_labels_GT.to(device)
        # forward
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y = one_hot_embedding(labels, K, device)
            y = y.to(device)
            loss, loss_first, loss_second = criterion(
                                outputs, y.float(), epoch, K, 
                                None, 0, None, None, 
                                kl_reg=False, 
                                device=device)

        # statistics
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects_GT += torch.sum(preds == labels.data)
    
    # print(f"##### length of datasets at phase {phase}: {len(dataloader.dataset)}") #pass 
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc_GT = running_corrects_GT / len(dataloader.dataset)
    epoch_acc_GT = epoch_acc_GT.cpu().item()
    return epoch_acc_GT, epoch_loss


def train_ENN(
    model,
    mydata,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device=None,
    logdir = "./",
):
    wandb.watch(model, log="all", log_freq=100)
    K = mydata.num_classes
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        begin_epoch = time.time()
        # Each epoch has a training and validation phase
        print("Training...")
        print(f" get last lr:{scheduler.get_last_lr()}") if scheduler else ""
        model.train()  # Set model to training mode
        dataloader = mydata.train_loader 

        running_loss = 0.0
        running_corrects = 0.0
        running_loss_GT = 0.0
        running_corrects_GT = 0.0

            # Iterate over data.
        for batch_idx, (inputs, single_labels_GT, labels) in enumerate(dataloader):
            # zero the parameter gradients
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            single_labels_GT = single_labels_GT.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y = one_hot_embedding(labels, K, device)
            y = y.to(device)
            loss, loss_first, loss_second = criterion(
                                outputs, y.float(), epoch, K, 
                                None, 0, None, None, 
                                kl_reg=False, 
                                device=device)
            loss.backward()
            optimizer.step()
            # statistics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects_GT += torch.sum(preds == single_labels_GT.data)
            running_corrects += torch.sum(preds == labels.data)
        if scheduler is not None:
            scheduler.step()

        # print(f"##### length of datasets at phase {phase}: {len(dataloader.dataset)}") #pass 
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects / len(dataloader.dataset)
        epoch_acc = epoch_acc.cpu().item()
        epoch_acc_GT = running_corrects_GT / len(dataloader.dataset)
        epoch_acc_GT = epoch_acc_GT.cpu().item()

        train_log("train", epoch, epoch_acc, epoch_acc_GT, epoch_loss)
        time_epoch_train = time.time() - begin_epoch
        print(
        f"Finish the Train in this epoch in {time_epoch_train//60:.0f}m {time_epoch_train%60:.0f}s.")

        #validation
        acc_GT_val, loss_val = validate(
            model, mydata.valid_loader, criterion,
            K, epoch)
        train_log("valid", epoch, 0, acc_GT_val, loss_val)
        
        if acc_GT_val > best_acc:
            best_acc = acc_GT_val
            best_epoch = epoch
            print(f"The best epoch: {best_epoch}, acc: {best_acc}")
            best_model_wts = copy.deepcopy(model.state_dict()) # deep copy the model
        
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


def make(args):
    mydata = None
    num_singles = 0
    num_comps = 0
    use_uncertainty = args.uncertainty
    milestone1 = args.milestone1
    milestone2 = args.milestone2
    
    if args.dataset == "tinyimagenet":
        mydata = tinyImageNetVague(
            args.data_dir, 
            num_comp=args.num_comp, 
            batch_size=args.batch_size,
            imagenet_hierarchy_path=args.data_dir,
            duplicate=True) #key duplicate 

    elif args.dataset == "cifar100":
        mydata = CIFAR100Vague(
            args.data_dir, 
            num_comp=args.num_comp,
            batch_size=args.batch_size,
            duplicate=True) #key duplicate

    num_singles = mydata.num_classes
    num_comps = mydata.num_comp
    print(f"Data: {args.dataset}, num of singleton and composite classes: {num_singles, num_comps}")

    if use_uncertainty:
        print("# use softplus activated model")
        if args.backbone == "EfficientNet-b3":
            model = HENN_EfficientNet(num_singles)
        else:
            print(f"### ERROR: The backbone {args.backbone} is invalid!")
    else:
        print("# use regular model without activation (softmax will be used later")
        if args.backbone == "EfficientNet-b3":
            model = EfficientNet_pretrain(num_singles)
        else:
            print(f"### ERROR: The backbone {args.backbone} is invalid!")
    model = model.to(device)

    if use_uncertainty:
        # if args.digamma:
        #     print("### Loss type: edl_digamma_loss")
        #     criterion = edl_digamma_loss
        # elif args.log:
        #     print("### Loss type: edl_log_loss")
        #     criterion = edl_log_loss
        # elif args.mse:
        print("### Loss type: edl_mse_loss")
        criterion = edl_mse_loss
        # else:
        #     print("ERROR: --uncertainty requires --mse, --log or --digamma.")
    else:
        print("### Loss type: CrossEntropy (no uncertainty)")
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone1, milestone2], gamma=0.1)
    return mydata, model, criterion, optimizer, scheduler


def main():
    set_random_seeds(args.seed)

    mydata, model, criterion, optimizer, scheduler = make(args)
    num_singles = mydata.num_classes

    if args.train:
        start = time.time()
        model, model_best, epoch_best = train_ENN(
                                            model,
                                            mydata,
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
        
        saved_path = os.path.join(base_path, "model_uncertainty_mse.pt")
        
        # saved_path = os.path.join(base_path, "model_CrossEntropy.pt")
        
        torch.save(state, saved_path)
        print(f"Saved: {saved_path}")
        end = time.time()
        print(f'Total training time for ENN: {(end-start)//60:.0f}m {(end-start)%60:.0f}s')
    else:
        print(f"## No training, load trained model directly")

    if args.test:
        valid_loader = mydata.valid_loader
        test_loader = mydata.test_loader
        R = mydata.R
        
        use_uncertainty = args.uncertainty
        if use_uncertainty:
            # if args.digamma:
            #     saved_path = os.path.join(base_path, "model_uncertainty_digamma.pt")
            # if args.log:
            #     saved_path = os.path.join(base_path, "model_uncertainty_log.pt")
            # if args.mse:
            saved_path = os.path.join(base_path, "model_uncertainty_mse.pt")
        else:
            saved_path = os.path.join(base_path, "model_CrossEntropy.pt")

        checkpoint = torch.load(saved_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        model_best_from_valid = copy.deepcopy(model)
        model_best_from_valid.load_state_dict(checkpoint["model_state_dict_best"]) 

        # model after the final epoch
        if args.vaguePred:
            js_result, prec_recall_f, js_comp, js_singl = evaluate_vague_final(
                model, test_loader, valid_loader, R, num_singles, device, 
                detNN=False)
        if args.nonVaguePred:
            acc_nonvague = evaluate_nonvague_final(model, test_loader, device)
        test_result_log(js_result, prec_recall_f, acc_nonvague, js_comp, js_singl, False) # bestModel=False

        print(f"### Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:\n")
        if args.vaguePred:
            js_result, prec_recall_f, js_comp, js_singl = evaluate_vague_final(
                model_best_from_valid, test_loader, valid_loader, R, num_singles,device,
                detNN=False)
        if args.nonVaguePred:
            acc_nonvague = evaluate_nonvague_final(model_best_from_valid, test_loader, device)
        test_result_log(js_result, prec_recall_f, acc_nonvague, js_comp, js_singl, True)


if __name__ == "__main__":
    # tell wandb to get started
    print(config)
    with wandb.init(project=f"{config['dataset']}-{config['num_comp']}M-ENN", config=config):
        config = wandb.config
        main()
