import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import copy
import time
import yaml
import wandb
import torch
torch.set_num_threads(4)
from torch import optim

from config_args import parser
from common_tools import create_path, set_device, set_random_seeds
from data.tinyImageNet import tinyImageNetOrig
from backbones import HENN_EfficientNet
from backbones import HENN_ResNet50, HENN_VGG16, HENN_LeNet, HENN_ResNet18
from helper_functions import one_hot_embedding
from loss import edl_mse_loss, edl_digamma_loss, edl_log_loss


def train_valid_log(phase, epoch, accGT, loss, loss_1, loss_2):
    wandb.log({
        f"{phase}_epoch": epoch, 
        f"{phase}_loss": loss, 
        f"{phase}_loss_1": loss_1, 
        f"{phase}_loss_2": loss_2, 
        f"{phase}_accGT": accGT}, step=epoch)
    print(f"{phase.capitalize()} loss: {loss:.4f}, loss1: {loss_1:.4f} , loss2: {loss_2:.4f},  accGT: {accGT:.4f}")


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


def validate(model, dataloader, criterion, K, epoch, entropy_lam, device):
    print("Validating...")
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_loss_1, running_loss_2 = 0.0, 0.0
    running_corrects = 0.0
    dataset_size_val = len(dataloader.dataset)
    for batch_idx, (inputs, single_labels_GT) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        labels = single_labels_GT.to(device, non_blocking=True)
        # forward
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y = one_hot_embedding(labels, K, device)
            # loss, loss_first, loss_second = criterion(
            #                     outputs, y, epoch, K, 
            #                     None, 0, None, None, 
            #                     kl_reg=False, 
            #                     device=device)

            loss, loss_first, loss_second = criterion(
                    outputs, y, epoch, K, 
                    None, 0, None, entropy_lam, None, None, None,
                    kl_reg=False, entropy_reg=True,
                    exp_type=5,
                    device=device)

        # statistics
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels)

        running_loss_1 += loss_first * batch_size
        running_loss_2 += loss_second * batch_size

    epoch_loss = running_loss / dataset_size_val
    epoch_acc = running_corrects / dataset_size_val
    epoch_acc = epoch_acc.detach()
    
    epoch_loss_1 = running_loss_1 / dataset_size_val
    epoch_loss_2 = running_loss_2 / dataset_size_val
    
    return epoch_acc, epoch_loss, epoch_loss_1, epoch_loss_2



def validate_test(model, dataloader, device, bestModel=False):
    print("Validating Test set...")
    model.eval()  # Set model to evaluate mode
    running_corrects = 0.0
    dataset_size_val = len(dataloader.dataset)
    for batch_idx, (inputs, single_labels_GT) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        labels = single_labels_GT.to(device, non_blocking=True)
        # forward
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels)

    epoch_acc = running_corrects / dataset_size_val
    epoch_acc = epoch_acc.detach()
    
    test_result_log(epoch_acc, bestModel=bestModel)


def train_ENN(
    model,
    mydata,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    entropy_lam=0.1,
    device=None,
    logdir = "./",
):
    wandb.watch(model, log="all", log_freq=100)
    since = time.time()
    K = mydata.num_classes
    dataloader = mydata.train_loader 
    dataset_size_train = len(dataloader.dataset)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(num_epochs):
        begin_epoch = time.time()
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        # Each epoch has a training and validation phase
        print("Training...")
        print(f" get last lr:{scheduler.get_last_lr()}") if scheduler else ""
        
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_loss_1, running_loss_2 = 0.0, 0.0
        
        running_corrects = 0.0

        # Iterate over data.
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y = one_hot_embedding(labels, K, device)
            # loss, loss_first, loss_second = criterion(
            #                     outputs, y, epoch, K, 
            #                     None, 0, None, None, 
            #                     kl_reg=False, 
            #                     device=device)
            loss, loss_first, loss_second = criterion(
                    outputs, y, epoch, K, 
                    None, 0, None, entropy_lam, None, None, None,
                    kl_reg=False, entropy_reg=True,
                    exp_type=5,
                    device=device)
            loss.backward()
            optimizer.step()
            
            # statistics
            batch_size = inputs.size(0)
            running_loss += loss.detach() * batch_size
            running_corrects += torch.sum(preds == labels)
        
            running_loss_1 += loss_first * batch_size
            running_loss_2 += loss_second * batch_size
        
        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / dataset_size_train
        epoch_acc = running_corrects / dataset_size_train
        epoch_acc = epoch_acc.detach()

        epoch_loss_1 = running_loss_1 / dataset_size_train
        epoch_loss_2 = running_loss_2 / dataset_size_train
        
        train_valid_log("train", epoch, epoch_acc, epoch_loss, epoch_loss_1, epoch_loss_2)
        time_epoch_train = time.time() - begin_epoch
        print(
        f"Finish the Train in this epoch in {time_epoch_train//60:.0f}m {time_epoch_train%60:.0f}s.")

        #validation phase
        valid_acc, valid_loss, valid_run_loss_1, valid_run_loss_2 = validate(
            model, mydata.valid_loader, criterion,
            K, epoch, entropy_lam, device)
        train_valid_log("valid", epoch, valid_acc, valid_loss, valid_run_loss_1, valid_run_loss_2)
        
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

    print("# use softplus activated model")
    if args.backbone == "EfficientNet-b3":
        model = HENN_EfficientNet(num_singles, pretrain=args.pretrain)
    elif args.backbone == "ResNet50":
        model = HENN_ResNet50(num_singles)
    elif args.backbone == "ResNet18":
        model = HENN_ResNet18(num_singles, pretrain=args.pretrain)
    elif args.backbone == "VGG16":
        model = HENN_VGG16(num_singles)
    elif args.backbone == "LeNet":
        model = HENN_LeNet(num_singles)
    else:
        print(f"### ERROR: The backbone {args.backbone} is invalid!")

    model = model.to(device)
    print("### Loss type: edl_digamma_loss")
    criterion = edl_digamma_loss

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
    tag0 = "_".join([f"{num_comp}M",f"Seed{seed}",f"BB{args.backbone}", "sweep_ENN"])
    base_path_spec_hyper_0 = os.path.join(base_path, tag0)
    create_path(base_path_spec_hyper_0)
    tag = "_".join([f"lr{init_lr}", f"EntrLam{args.entropy_lam}"])
    base_path_spec_hyper = os.path.join(base_path_spec_hyper_0, tag)
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
            model, model_best, epoch_best = train_ENN(
                                                model,
                                                mydata,
                                                criterion,
                                                optimizer,
                                                scheduler=scheduler,
                                                num_epochs=args.epochs,
                                                entropy_lam=args.entropy_lam,
                                                device=device,
                                                )
            state = {
                "epoch_best": epoch_best,
                "model_state_dict": model.state_dict(),
                "model_state_dict_best": model_best.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            
            saved_path = os.path.join(base_path_spec_hyper, "model_uncertainty_digamma.pt")
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

            # if args.digamma:
            #     saved_path = os.path.join(base_path, "model_uncertainty_digamma.pt")
            # if args.log:
            #     saved_path = os.path.join(base_path, "model_uncertainty_log.pt")
            # if args.mse:
            saved_path = os.path.join(base_path_spec_hyper, "model_uncertainty_digamma.pt")

            checkpoint = torch.load(saved_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

            model_best_from_valid = copy.deepcopy(model)
            model_best_from_valid.load_state_dict(checkpoint["model_state_dict_best"]) 

            # model after the final epoch
            print(f"\n### Evaluate the model after all epochs:")
            validate_test(model, test_loader, device, bestModel=False)

            print(f"\n### Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:\n")
            validate_test(model_best_from_valid, test_loader, device, bestModel=True)


if __name__ == "__main__":
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
    
    # config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    # opt.update(config)
    # args = opt

    # # convert args from Dict to Object
    # args = dictToObj(args)
    # args.device = set_device(args.gpu)

    # # tell wandb to get started
    # print(config)
    # with wandb.init(project=f"{config['dataset']}-{config['num_comp']}M-ENN", config=config):
    #     config = wandb.config
    #     main(args)
    
    project_name = "CIFAR100-5M-Ker3-ENN-sweep"
    # main(project_name, opt)
    sweep_id = "ai5o0sh8"
    entity = "changbinli"
    # wandb.agent(sweep_id, function=main(project_name, opt), entity=entity, project=project_name, count=1)
    main(project_name, opt)
