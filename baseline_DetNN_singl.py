import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import copy
import time 
import yaml
import wandb
import torch
torch.set_num_threads(4)
from torch import optim, nn
import torch.nn.functional as F
from collections import Counter

from config_args import parser
from common_tools import create_path, set_device, dictToObj, set_random_seeds
from data.tinyImageNet import tinyImageNetOrig
from backbones import EfficientNet_pretrain, ResNet50, ResNet18, VGG16, LeNet


def train_valid_log(phase, epoch, accGT, loss):
    wandb.log({
        f"{phase}_epoch": epoch, 
        f"{phase}_loss": loss, 
        f"{phase}_accGT": accGT}, step=epoch)
    print(f"{phase.capitalize()} loss: {loss:.4f} accGT: {accGT:.4f}")


def test_result_log(
    nonvague_acc, 
    bestModel=False):
    if bestModel:
        tag = "TestB"
    else:
        tag = "TestF"
    wandb.log({
        f"{tag} accNonVague": nonvague_acc})
    print(f"{tag} accNonVague: {nonvague_acc:.4f},\n")


def validate(model, dataloader, criterion, device):
    print("Validating...")
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0.0
    dataset_size_val = len(dataloader.dataset)
    for batch_idx, (inputs, single_labels_GT) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        labels = single_labels_GT.to(device, non_blocking=True)
        # forward
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / dataset_size_val
    epoch_acc = running_corrects / dataset_size_val
    epoch_acc = epoch_acc.detach()
    return epoch_acc, epoch_loss


def train_DetNN(
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
    since = time.time()
    dataloader = mydata.train_loader 
    dataset_size_train = len(dataloader.dataset)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(num_epochs):
        begin_epoch = time.time()
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        print("Training...")
        print(f" get last lr:{scheduler.get_last_lr()}") if scheduler else ""

        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0.0
        epoch_acc, epoch_acc_GT = 0.0, 0.0
        # Iterate over data.
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            batch_size = inputs.size(0)
            running_loss += loss.detach() * batch_size
            running_corrects += torch.sum(preds == labels)

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / dataset_size_train
        epoch_acc = running_corrects / dataset_size_train
        epoch_acc = epoch_acc.detach()
        
        train_valid_log("train", epoch, epoch_acc, epoch_loss)
        time_epoch_train = time.time() - begin_epoch
        print(
        f"Finish the Train in this epoch in {time_epoch_train//60:.0f}m {time_epoch_train%60:.0f}s.")

        # Validation phase
        valid_acc, valid_loss = validate(model, mydata.valid_loader, criterion, device)
        train_valid_log("valid", epoch, valid_acc, valid_loss)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            wandb.run.summary["best_valid_acc"] = valid_acc
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




def make(args):
    mydata = None
    num_singles = 0
    num_comps = 0
    milestone1 = args.milestone1
    milestone2 = args.milestone2
    device = args.device
    
    if args.dataset == "tinyimagenet":
        mydata = tinyImageNetOrig(
            args.data_dir, 
            batch_size=args.batch_size,
            num_workers=args.num_workers
            )
        
    num_singles = mydata.num_classes
    num_comps = mydata.num_comp
    print(f"Data: {args.dataset}, num of singleton and composite classes: {num_singles, num_comps}")
    
    if args.backbone == "EfficientNet-b3":
        model = EfficientNet_pretrain(num_singles, pretrain=args.pretrain)
    elif args.backbone == "ResNet50":
        model = ResNet50(num_singles)
    elif args.backbone == "ResNet18":
        model = ResNet18(num_singles, pretrain=args.pretrain)
    elif args.backbone == "VGG16":
        model = VGG16(num_singles)
    elif args.backbone == "LeNet":
        model = LeNet(num_singles)
    else:
        print(f"### ERROR: The backbone {args.backbone} is invalid!")
    model = model.to(device)
    print("### Loss type: CrossEntropy (no uncertainty)")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone1, milestone2], gamma=0.1)
    return mydata, model, criterion, optimizer, scheduler


def generateSpecPath(args):
    output_folder=args.output_folder
    saved_spec_dir=args.saved_spec_dir 
    num_comp=args.num_comp
    init_lr=args.init_lr
    seed=args.seed
    
    base_path = os.path.join(output_folder, saved_spec_dir)
    tag0 = "_".join([f"{num_comp}M", f"Seed{seed}", f"BB{args.backbone}", "sweep_DNN"])
    base_path_spec_hyper_0 = os.path.join(base_path, tag0)
    create_path(base_path_spec_hyper_0)
    base_path_spec_hyper = os.path.join(base_path_spec_hyper_0, str(init_lr))
    create_path(base_path_spec_hyper)
    return base_path_spec_hyper

def main(project_name, args_all):
    # Initialize a new wandb run
    with wandb.init(project=project_name, config=args_all):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        # wandb.config has the lastest parameters
        args = wandb.config
        print(f"Current wandb.config: {wandb.config}")
        # create a more specfic path to save the model for the current hyperparameter
        base_path_spec_hyper = generateSpecPath(args)
        set_random_seeds(args.seed)
        device = args.device
        mydata, model, criterion, optimizer, scheduler = make(args)
        num_singles = mydata.num_classes
        
        if args.train:
            start = time.time()
            model, model_best, epoch_best = train_DetNN(
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
            saved_path = os.path.join(base_path_spec_hyper, "model_CrossEntropy.pt")
            torch.save(state, saved_path)
            print(f"Saved: {saved_path}")
            end = time.time()
            print(f'Total training time for DNN: {(end-start)//60:.0f}m {(end-start)%60:.0f}s')
        else:
            print(f"## No training, load trained model directly")
        
        if args.test:
            test_loader = mydata.test_loader
            saved_path = os.path.join(base_path_spec_hyper, "model_CrossEntropy.pt")
            checkpoint = torch.load(saved_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

            model_best_from_valid = copy.deepcopy(model)
            model_best_from_valid.load_state_dict(checkpoint["model_state_dict_best"]) 

            # model after the final epoch
            # bestModel=False
            print(f"\n### Evaluate the model after all epochs:")
            epoch_acc, _ = validate(model, test_loader, criterion, device)            
            test_result_log(epoch_acc, bestModel=False)

            print(f"\n### Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:\n")
            epoch_acc, _ = validate(model_best_from_valid, test_loader, criterion, device)            
            test_result_log(epoch_acc, bestModel=True)

if __name__ == "__main__":
    # https://github.com/wandb/sweeps/blob/master/examples/nested-params/train-nested-py.py
    # https://docs.wandb.ai/guides/track/log?_gl=1*sutwuo*_ga*MjA4NDU0MTg1My4xNjY3NDI5MDk5*_ga_JH1SJHJQXJ*MTY3Mjg2Mzk4NC45Mi4xLjE2NzI4NjUxMjguMzguMC4w#summary-metrics
    
    args = parser.parse_args()
    opt = vars(args)
    # build the path to save model and results
    create_path(args.output_folder)
    base_path = os.path.join(args.output_folder, args.saved_spec_dir)
    create_path(base_path)
    config_file = os.path.join(base_path, "config.yml")
    # A user-specified nested config.
    CONFIG = yaml.load(open(config_file), Loader=yaml.FullLoader)
    opt.update(CONFIG)
    # convert args from Dict to Object
    # args = dictToObj(opt)
    opt["device"] = set_device(args.gpu)

    # tell wandb to get started
    print("All hyperparameters:", opt)
    # with wandb.init(project=f"{CONFIG['dataset']}-{CONFIG['num_comp']}M-DNN", config=CONFIG):
    # with wandb.init(config=CONFIG):
        # config = wandb.config
    
    project_name = "CIFAR100-5M-Ker3-DNN-sweep"
    # main(project_name, opt)
    sweep_id = "ai5o0sh8"
    entity = "changbinli"
    # wandb.agent(sweep_id, function=main(project_name, opt), entity=entity, project=project_name, count=1)
    main(project_name, opt)
