# Import libraries
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import time
import yaml
import wandb 
import copy 
import torch
torch.set_num_threads(4)
from torch import optim

from config_args import parser  
from common_tools import create_path, set_device, dictToObj, set_random_seeds
from data.cifar10h import CIFAR10h
from data.tinyImageNet import tinyImageNetVague
from data.breeds import BREEDSVague
from data.mnist import MNIST
from backbones import HENN_EfficientNet, HENN_ResNet50, HENN_VGG16, HENN_LeNet, HENN_LeNet_v2, HENN_ResNet18
# from backbones import EfficientNet_pretrain, ResNet50
from GDD_train import train_model
from GDD_test import evaluate_vague_nonvague
from loss import edl_mse_loss, edl_digamma_loss, edl_log_loss
from loss import henn_gdd, unified_UCE_loss


def make(args):
    mydata = None
    num_singles = 0
    num_comps = 0
    num_classes_both = 0 
    
    ### Dataset ###
    if args.dataset == "CIFAR10h":
        mydata = CIFAR10h(
            args.data_dir, 
            batch_size=args.batch_size,
            pretrain=args.pretrain,
            num_workers=args.num_workers,
            seed=args.seed,
            )

    num_singles = mydata.num_classes
    num_comps = mydata.num_comp
    print(f"Data: {args.dataset}, num of singleton and composite classes: {num_singles, num_comps}")
    
    ### Backbone ###
    num_classes_both = num_singles + num_comps
    if args.backbone == "EfficientNet-b3":
        model = HENN_EfficientNet(num_classes_both, pretrain=args.pretrain)
    elif args.backbone == "ResNet50":
        model = HENN_ResNet50(num_classes_both)
    elif args.backbone == "ResNet18":
        model = HENN_ResNet18(num_classes_both)
    elif args.backbone == "VGG16":
        model = HENN_VGG16(num_classes_both)
    elif args.backbone == "LeNet":
        model = HENN_LeNet(num_classes_both)
    elif args.backbone == "LeNetV2":
        model = HENN_LeNet_v2(out_dim=num_classes_both)
    else:
        print(f"### ERROR {args.dataset}: The backbone {args.backbone} is invalid!")
    model = model.to(args.device)

    ### Loss ###
    if args.digamma:
        print("### Loss type: edl_digamma_loss")
        criterion = edl_digamma_loss
    elif args.henn_gdd:
        print("### Loss type: GDD")
        # criterion = henn_gdd
        criterion = unified_UCE_loss
    else:
        parser.error("--uncertainty requires --mse, --log or --digamma.")

    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.init_lr, weight_decay=args.wd, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=0.005)
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # if args.pretrain:
        # exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.milestone1, args.milestone2], gamma=0.1)
    return mydata, model, criterion, optimizer, scheduler


def generateSpecPath(args):
    base_path = os.path.join(args.output_folder, args.saved_spec_dir)
    tag0 = "_".join([f"SEED{args.seed}", f"BB{args.backbone}", f"{args.num_comp}M", f"Ker{args.gauss_kernel_size}", "sweep", f"GDDexp{args.exp_type}"])
    tag = "_".join(["lr", str(args.init_lr), "klLamGDD", str(args.kl_lam_GDD), "EntrLamDir", str(args.entropy_lam_Dir), "EntrLamGDD", str(args.entropy_lam_GDD)])
    base_path_spec_hyper_0 = os.path.join(base_path, tag0)
    create_path(base_path_spec_hyper_0)
    base_path_spec_hyper = os.path.join(base_path_spec_hyper_0, tag)
    create_path(base_path_spec_hyper)
    return base_path_spec_hyper


def main(args):
    print(f"Current all hyperparameters: {args}")
    base_path_spec_hyper = generateSpecPath(args)    
    print(f"Model: Train:{args.train}, Test: {args.test}")
    set_random_seeds(args.seed)
    device = args.device
    mydata, model, criterion, optimizer, scheduler = make(args)
    num_singles = mydata.num_classes
    num_classes = num_singles + mydata.num_comp
    print("Total number of classes to train: ", num_classes)

    # saved path for model
    if args.digamma:
        saved_path = os.path.join(base_path_spec_hyper, "model_uncertainty_digamma.pt")
    if args.henn_gdd:
        saved_path = os.path.join(base_path_spec_hyper, "model_uncertainty_gdd.pt")

    if args.train:
        start = time.time()
        model, model_best, epoch_best, model_best_GT, epoch_best_GT = train_model(
            args,
            model,
            mydata,
            criterion,
            optimizer,
            scheduler=scheduler,
            device=device,
            )

        state = {
            "epoch_best": epoch_best,
            "model_state_dict": model.state_dict(),
            "model_state_dict_best": model_best.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch_best_GT": epoch_best_GT,
            "model_state_dict_best_GT": model_best_GT.state_dict(),
        }

        torch.save(state, saved_path)
        print(f"Saved: {saved_path}")
        end = time.time()
        print(f'Total training time for HENN: %s seconds.'%str(end-start))
    else:
        print(f"## No training, load trained model directly")

    if args.test:
        checkpoint = torch.load(saved_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        model_best_from_valid = copy.deepcopy(model)
        model_best_from_valid.load_state_dict(checkpoint["model_state_dict_best"]) 
        
        model_best_from_valid_GT = copy.deepcopy(model)
        model_best_from_valid_GT.load_state_dict(checkpoint["model_state_dict_best_GT"]) 

        #! Test set
        # #Evaluation, Inference
        print(f"\n### Evaluate the model after all epochs:")
        evaluate_vague_nonvague(
            model, mydata.test_loader, mydata.R, 
            mydata.num_classes, mydata.num_comp, mydata.vague_classes_ids,
            None, device, train_flag=3)

        print(f"\n### Use the model selected from ValidSet in Ep. {checkpoint['epoch_best']}:")
        evaluate_vague_nonvague(
            model_best_from_valid, mydata.test_loader, mydata.R, 
            mydata.num_classes, mydata.num_comp, mydata.vague_classes_ids,
            None, device, bestModel=True, train_flag=3)

        print(f"\n### Use the model selected from ValidSet (GT) in Ep. {checkpoint['epoch_best_GT']}:")
        evaluate_vague_nonvague(
            model_best_from_valid_GT, mydata.test_loader, mydata.R, 
            mydata.num_classes, mydata.num_comp, mydata.vague_classes_ids,
            None, device, bestModelGT=True, train_flag=3)

        # #! Training set for debugging
        # #Evaluation, Inference
        print(f"\n### (TrainingSet) Evaluate the model after all epochs:")
        evaluate_vague_nonvague(
            model, mydata.train_loader, mydata.R, 
            mydata.num_classes, mydata.num_comp, mydata.vague_classes_ids,
            None, device, train_flag=1)

        print(f"\n### (TrainingSet) Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:")
        evaluate_vague_nonvague(
            model_best_from_valid, mydata.train_loader, mydata.R, 
            mydata.num_classes, mydata.num_comp, mydata.vague_classes_ids,
            None, device, bestModel=True, train_flag=1)


if __name__ == "__main__":
    args = parser.parse_args()
    opt = vars(args)

    # build the path to save model and results
    create_path(args.output_folder) 
    base_path = os.path.join(args.output_folder, args.saved_spec_dir)
    create_path(base_path)

    config_file = os.path.join(base_path, "config_GDD_relabeled.yml")
    CONFIG = yaml.load(open(config_file), Loader=yaml.FullLoader)
    opt.update(CONFIG)

    # else:  # yaml priority is higher than args
    #     opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    #     opt.update(vars(args))
    #     args = argparse.Namespace(**opt)

    # convert args from Dict to Object
    # args = dictToObj(opt)
    opt["device"] = set_device(args.gpu)

    # tell wandb to get started
    print("Default setting before hyperparameters tuning:", opt)
    project_name = f"{opt['dataset']}-{opt['num_comp']}M-Ker{opt['gauss_kernel_size']}-HENNgdd-Debug"
    with wandb.init(project=project_name, config=opt):
        config = wandb.config
        main(config)
