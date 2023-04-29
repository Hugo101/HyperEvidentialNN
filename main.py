# Import libraries
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import numpy as np
import pandas as pd
from datetime import datetime as dt
import time
import yaml       
from sklearn import metrics
import urllib
import zipfile
import wandb 
import copy 
import torch
torch.set_num_threads(4)
from torch import optim, nn
import torch.nn.functional as F

from config_args import parser  
from common_tools import create_path, set_device, dictToObj, set_random_seeds
from data.tinyImageNet import tinyImageNetVague
from data.cifar100 import CIFAR100Vague
from data.breeds import BREEDSVague
from data.mnist import MNIST
from backbones import HENN_EfficientNet, HENN_ResNet50, HENN_VGG16, HENN_LeNet
# from backbones import EfficientNet_pretrain, ResNet50
from train import train_model
from test import evaluate_vague_nonvague
from loss import edl_mse_loss, edl_digamma_loss, edl_log_loss


def make(args):
    mydata = None
    num_singles = 0
    num_comps = 0
    num_classes_both = 0 
    use_uncertainty = args.use_uncertainty
    milestone1 = args.milestone1
    milestone2 = args.milestone2
    
    if args.dataset == "tinyimagenet":
        mydata = tinyImageNetVague(
            args.data_dir, 
            num_comp=args.num_comp, 
            batch_size=args.batch_size,
            imagenet_hierarchy_path=args.data_dir,
            blur=args.blur,
            gray=args.gray,
            gauss_kernel_size=args.gauss_kernel_size,
            pretrain=args.pretrain,
            num_workers=args.num_workers,
            seed=args.seed)

    elif args.dataset == "cifar100":
        mydata = CIFAR100Vague(
            args.data_dir, 
            num_comp=args.num_comp, 
            batch_size=args.batch_size,
            blur=args.blur,
            gauss_kernel_size=args.gauss_kernel_size,
            pretrain=args.pretrain,
            num_workers=args.num_workers,
            seed=args.seed,
            comp_el_size=args.num_subclasses,
            )

    elif args.dataset in ["living17", "nonliving26", "entity13", "entity30"]:
        data_path_base = os.path.join(args.data_dir, "ILSVRC/ILSVRC")
        mydata = BREEDSVague(
            os.path.join(data_path_base, "BREEDS/"),
            os.path.join(data_path_base, 'Data', 'CLS-LOC/'),
            ds_name=args.dataset,
            num_comp=args.num_comp, 
            batch_size=args.batch_size,
            blur=args.blur,
            gauss_kernel_size=args.gauss_kernel_size,
            pretrain=args.pretrain,
            num_workers=args.num_workers,
            seed=args.seed,
            comp_el_size=args.num_subclasses,
            )

    elif args.dataset == "mnist":
        mydata = MNIST(
            args.data_dir,
            batch_size=args.batch_size,
            blur=args.blur,
            gauss_kernel_size=args.gauss_kernel_size,
            pretrain=args.pretrain,
            num_workers=args.num_workers,
            seed=args.seed,
            )

    num_singles = mydata.num_classes
    num_comps = mydata.num_comp
    print(f"Data: {args.dataset}, num of singleton and composite classes: {num_singles, num_comps}")
    
    num_classes_both = num_singles + num_comps
    if args.backbone == "EfficientNet-b3":
        model = HENN_EfficientNet(num_classes_both, pretrain=args.pretrain)
    elif args.backbone == "ResNet50":
        model = HENN_ResNet50(num_classes_both)
    elif args.backbone == "VGG16":
        model = HENN_VGG16(num_classes_both)
    elif args.backbone == "LeNet":
        model = HENN_LeNet(num_classes_both)
    else:
        print(f"### ERROR {args.dataset}: The backbone {args.backbone} is invalid!")

    model = model.to(args.device)

    if use_uncertainty:
        if args.digamma:
            print("### Loss type: edl_digamma_loss")
            criterion = edl_digamma_loss
        elif args.log:
            print("### Loss type: edl_log_loss")
            criterion = edl_log_loss
        elif args.mse:
            print("### Loss type: edl_mse_loss")
            criterion = edl_mse_loss
        else:
            parser.error("--uncertainty requires --mse, --log or --digamma.")
    else:
        print("### Loss type: CrossEntropy (no uncertainty)")
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=0.005)
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # if args.pretrain:
        # exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone1, milestone2], gamma=0.1)
    return mydata, model, criterion, optimizer, scheduler


def generateSpecPath(
    exp_type, 
    output_folder, saved_spec_dir, 
    num_comp,
    gauss_kernel_size,
    init_lr, 
    kl_lam, 
    kl_lam_teacher, forward_kl_teacher, 
    entropy_lam, 
    ce_lam):
    base_path = os.path.join(output_folder, saved_spec_dir)
    if exp_type in [5, 6]:
        tag0 = "_".join([f"{num_comp}M", f"ker{gauss_kernel_size}", "sweep", f"HENNexp{exp_type}"])
        tag = "_".join(["lr", str(init_lr), "EntropyLam", str(entropy_lam)])
    base_path_spec_hyper_0 = os.path.join(base_path, tag0)
    create_path(base_path_spec_hyper_0)
    base_path_spec_hyper = os.path.join(base_path_spec_hyper_0, tag)
    create_path(base_path_spec_hyper)
    return base_path_spec_hyper


def main(args):
    print(f"Current all hyperparameters: {args}")
    base_path_spec_hyper = generateSpecPath(
        args.exp_type,
        args.output_folder, args.saved_spec_dir,
        args.num_comp,
        args.gauss_kernel_size, 
        args.init_lr, 
        args.kl_lam, 
        args.kl_lam_teacher, args.forward_kl_teacher, 
        args.entropy_lam, 
        args.ce_lam)
    
    print(f"Model: Train:{args.train}, Test: {args.test}")
    set_random_seeds(args.seed)
    device = args.device
    mydata, model, criterion, optimizer, scheduler = make(args)
    num_singles = mydata.num_classes
    num_comps = mydata.num_comp
    num_classes = num_singles + num_comps
    print("Total number of classes to train: ", num_classes)

    if args.train:
        start = time.time()
        model, model_best, epoch_best = train_model(
                                            model,
                                            mydata,
                                            num_classes,
                                            criterion,
                                            optimizer,
                                            scheduler=scheduler,
                                            num_epochs=args.epochs,
                                            uncertainty=args.use_uncertainty,
                                            kl_reg=args.kl_reg,
                                            kl_lam=args.kl_lam,
                                            kl_reg_teacher=args.kl_reg_teacher,
                                            kl_lam_teacher=args.kl_lam_teacher,
                                            forward_kl_teacher=args.forward_kl_teacher,
                                            saved_path_teacher=args.saved_path_teacher,
                                            entropy_reg=args.entropy_reg,
                                            entropy_lam=args.entropy_lam,
                                            ce_lam=args.ce_lam,
                                            exp_type=args.exp_type,
                                            device=device,
                                            logdir=base_path_spec_hyper,
                                            )

        state = {
            "epoch_best": epoch_best,
            "model_state_dict": model.state_dict(),
            "model_state_dict_best": model_best.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if args.use_uncertainty:
            if args.digamma:
                saved_path = os.path.join(base_path_spec_hyper, "model_uncertainty_digamma.pt")
            if args.log:
                saved_path = os.path.join(base_path_spec_hyper, "model_uncertainty_log.pt")
            if args.mse:
                saved_path = os.path.join(base_path_spec_hyper, "model_uncertainty_mse.pt")
        else:
            saved_path = os.path.join(base_path_spec_hyper, "model_CrossEntropy.pt")
        torch.save(state, saved_path)
        print(f"Saved: {saved_path}")
        end = time.time()
        print(f'Total training time for HENN: %s seconds.'%str(end-start))
    else:
        print(f"## No training, load trained model directly")

    if args.test:
        use_uncertainty = args.use_uncertainty
        if use_uncertainty:
            if args.digamma:
                saved_path = os.path.join(base_path_spec_hyper, "model_uncertainty_digamma.pt")
            if args.log:
                saved_path = os.path.join(base_path_spec_hyper, "model_uncertainty_log.pt")
            if args.mse:
                saved_path = os.path.join(base_path_spec_hyper, "model_uncertainty_mse.pt")
        else:
            saved_path = os.path.join(base_path_spec_hyper, "model_CrossEntropy.pt")
        
        checkpoint = torch.load(saved_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        model_best_from_valid = copy.deepcopy(model)
        model_best_from_valid.load_state_dict(checkpoint["model_state_dict_best"]) 

        # #Evaluation, Inference
        print(f"\n### Evaluate the model after all epochs:")
        evaluate_vague_nonvague(
            model, mydata.test_loader, mydata.R, 
            mydata.num_classes, mydata.num_comp, mydata.vague_classes_ids,
            None, device)

        print(f"\n### Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:")
        evaluate_vague_nonvague(
            model_best_from_valid, mydata.test_loader, mydata.R, 
            mydata.num_classes, mydata.num_comp, mydata.vague_classes_ids,
            None, device, bestModel=True)


if __name__ == "__main__":
    args = parser.parse_args()
    # process argparse & yaml
    # if  args.config:
    opt = vars(args)

    # build the path to save model and results
    create_path(args.output_folder) 
    base_path = os.path.join(args.output_folder, args.saved_spec_dir)
    create_path(base_path)

    config_file = os.path.join(base_path, "config.yml")
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
    with wandb.init(project=f"{opt['dataset']}-{opt['num_comp']}M-Ker{opt['gauss_kernel_size']}-HENN-Sweep", config=opt):
        config = wandb.config
        main(config)
