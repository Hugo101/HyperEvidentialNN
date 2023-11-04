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
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter

from config_args import parser
from common_tools import create_path, set_device, dictToObj, set_random_seeds
from data.tinyImageNet import tinyImageNetVague
from data.cifar100 import CIFAR100Vague
from data.breeds import BREEDSVague
from data.mnist import MNIST
from data.cifar10h import CIFAR10h
from backbones import EfficientNet_pretrain, ResNet50, ResNet18, VGG16, LeNet
from helper_functions import js_subset, acc_subset


def train_valid_log(phase, epoch, accDup, accGT, loss):
    wandb.log({
        f"{phase}_epoch": epoch, 
        f"{phase}_loss": loss, 
        f"{phase}_accDup": accDup, 
        f"{phase}_accGT": accGT}, step=epoch)
    print(f"{phase.capitalize()} loss: {loss:.4f} accDup: {accDup:.4f} accGT: {accGT:.4f}")


def test_result_log(
    js_result, prec_recall_f, js_comp, js_singl, 
    nonvague_acc, nonvague_acc_singl, 
    bestModel=False):
    if bestModel:
        tag = "TestB"
    else:
        tag = "TestF"
    wandb.log({
        f"{tag} JSoverall": js_result[0], 
        f"{tag} JScomp": js_result[1], 
        f"{tag} JSsngl": js_result[2],
        f"{tag} CmpPreci": prec_recall_f[0], 
        f"{tag} CmpRecal": prec_recall_f[1], 
        f"{tag} CmpFscor": prec_recall_f[2], 
        f"{tag} js_comp": js_comp,
        f"{tag} js_singl": js_singl,
        f"{tag} accNonVague": nonvague_acc,
        f"{tag} accNonVagueSingl": nonvague_acc_singl})
    print(f"{tag} accNonVague: {nonvague_acc:.4f},\n\
        accNonVagueSingl: {nonvague_acc_singl:.4f},\n \
        JS(O_V_N): {js_result}, P_R_F_compGTcnt_cmpPREDcnt: {prec_recall_f}\n")


def validate(model, dataloader, criterion, device):
    print("Validating...")
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0.0
    dataset_size_val = len(dataloader.dataset)
    for batch_idx, (inputs, single_labels_GT, single_label_dup) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        # labels = single_labels_GT.to(device, non_blocking=True)
        labels = single_label_dup.to(device, non_blocking=True) ### important
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
        running_corrects_GT = 0.0
        epoch_acc, epoch_acc_GT = 0.0, 0.0
        # Iterate over data.
        for batch_idx, (inputs, single_labels_GT, labels) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            single_labels_GT = single_labels_GT.to(device, non_blocking=True)
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
            running_corrects_GT += torch.sum(preds == single_labels_GT)

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / dataset_size_train
        epoch_acc = running_corrects / dataset_size_train
        epoch_acc = epoch_acc.detach()
        epoch_acc_GT = running_corrects_GT / dataset_size_train
        epoch_acc_GT = epoch_acc_GT.detach()
        
        train_valid_log("train", epoch, epoch_acc, epoch_acc_GT, epoch_loss)
        time_epoch_train = time.time() - begin_epoch
        print(
        f"Finish the Train in this epoch in {time_epoch_train//60:.0f}m {time_epoch_train%60:.0f}s.")

        # Validation phase
        valid_acc, valid_loss = validate(model, mydata.valid_loader, criterion, device)
        train_valid_log("valid", epoch, valid_acc, 0, valid_loss)
        
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


def num_accurate_baseline(output, labels, R, cutoff, detNN=True):
    if detNN:
        p_exp = F.softmax(output, dim=1)
    else:
        alpha = torch.add(output, 1)
        alpha_sum = torch.sum(alpha, dim=1)
        # Get the predicted labels
        p_exp = torch.div(alpha, alpha_sum[:, None])

    predicted_labels = torch.argmax(p_exp, dim=1)
    total_correct = 0.0
    for i in range(len(labels)):
        indices = (p_exp[i] >= cutoff).nonzero(as_tuple=True)[0]
        predicted_set = set(indices.tolist())

        if len(predicted_set) == 1:
            predicted_set = set(R[predicted_labels[i].item()])

        ground_truth_set = set(R[labels[i]])
        intersect = predicted_set.intersection(ground_truth_set)
        union = predicted_set.union(ground_truth_set)
        total_correct += float(len(intersect)) / len(union)
    return total_correct


@torch.no_grad()
def evaluate_cutoff(
    model, val_loader, R, 
    cutoff, device,
    detNN=True):
    model.eval()
    total_correct = 0.0
    total_samples = 0
    # losses = []
    for batch in val_loader:
        images, _, labels = batch
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        output = model(images)
        total_correct += num_accurate_baseline(output, labels, R, cutoff, detNN=detNN)
        total_samples += len(labels)
    accuracy = total_correct / total_samples
    return accuracy


def get_cutoff(
    model, val_loader, R, 
    device,
    detNN=True): #todo: could be better for efficiency
    cutoff = 0.0
    if detNN:
        end_cutoff = 0.5
        interval = 0.05
    else:
        end_cutoff = 0.05
        interval = 0.001
    accs = []
    cutoffs = []
    while cutoff <= end_cutoff:
        js = evaluate_cutoff(model, val_loader, R, cutoff, device, detNN=detNN)
        print(f"For cutoff = {cutoff:.3f}, Validation JS: {js:.4f}")    
        accs.append(js)
        cutoffs.append(cutoff)
        cutoff += interval
    maxID = torch.argmax(torch.tensor(accs)) 
    return cutoffs[maxID]


def calculate_metrics(output, labels, R, cutoff, detNN=True):
    if detNN:
        p_exp = F.softmax(output, dim=1)
    else:
        alpha = torch.add(output, 1)
        alpha_sum = torch.sum(alpha, dim=1)
        p_exp = torch.div(alpha, alpha_sum[:, None])
    
    # Get the predicted labels
    predicted_labels = torch.argmax(p_exp, dim=1)
    num_singles = output.shape[1]
    correct_vague = 0.0
    correct_nonvague = 0.0
    vague_total = 0
    nonvague_total = 0
    predSet_or_not = []
    for i in range(len(labels)):
        indices = (p_exp[i] >= cutoff).nonzero(as_tuple=True)[0]
        predicted_set = set(indices.tolist())

        if len(predicted_set) == 1:
            predicted_set = set(R[predicted_labels[i].item()])
            predSet_or_not.append(0) # singleton
        else:
            predSet_or_not.append(1)

        ground_truth_set = set(R[labels[i].item()])
        intersect = predicted_set.intersection(ground_truth_set)
        union = predicted_set.union(ground_truth_set)
        if len(predicted_set) == 1:
            correct_nonvague += float(len(intersect)) / len(union)
            nonvague_total += 1
        else:
            correct_vague += float(len(intersect)) / len(union)
            vague_total += 1
    stat_result = [correct_nonvague, correct_vague, nonvague_total, vague_total]
    
    predSet_or_not = torch.tensor(predSet_or_not) #1:vague, 0：non-vague
    # check precision, recall, f-score for composite classes
    prec_r_f = precision_recall_f_v1(labels, predSet_or_not, num_singles)
    return stat_result, prec_r_f


def precision_recall_f_v1(y_test, y_pred, num_singles):
    # make singleton labels 0, and composite labels 1
    y_test = y_test.cpu().numpy()
    y_test = y_test >= num_singles
    y_pred = y_pred.cpu().numpy()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    label_value_cnt = Counter(y_test)
    pred_value_cnt = Counter(y_pred)

    comp_GT_cnt = label_value_cnt[True]
    cmp_pred_cnt = pred_value_cnt[True]
    
    return precision, recall, f1, comp_GT_cnt, cmp_pred_cnt


@torch.no_grad()
def evaluate_vague_nonvague_final(
    model,
    test_loader,
    val_loader,
    R,
    num_singles,
    device,
    detNN=True,
    bestModel=False):
    cutoff = get_cutoff(model, val_loader, R, device, detNN=detNN)
    print(f"### selected cutoff: {cutoff}")
    
    # begin evaluation
    model.eval()
    outputs_all = []
    labels_all = []
    true_labels_all = []
    preds_all = []
    total_correct = 0.0
    total_samples = 0
    for batch in test_loader:
        images, single_labels_GT, labels = batch
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        single_labels_GT = single_labels_GT.to(device, non_blocking=True)
        output = model(images)
        preds = output.argmax(dim=1)
        total_correct += torch.sum(preds == single_labels_GT) # nonvague
        total_samples += len(labels)
        
        outputs_all.append(output)
        labels_all.append(labels)
        preds_all.append(preds)
        true_labels_all.append(single_labels_GT)
    print(f"Total samples in test set: {total_samples}")
    # nonvague prediction accuracy
    nonvague_acc = total_correct / total_samples  
    
    outputs_all = torch.cat(outputs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    preds_all = torch.cat(preds_all, dim=0)
    true_labels_all = torch.cat(true_labels_all, dim=0)
    
    # JS of composite examples (original, not prediction)
    comp_idx = labels_all > num_singles-1
    # acc_comp = acc_subset(comp_idx, labels_all, preds_all)
    js_comp = js_subset(comp_idx, labels_all, preds_all, R)
    # JS of singleton examples
    singl_idx = labels_all < num_singles
    # acc_singl = acc_subset(singl_idx, labels_all, preds_all)
    js_singl = js_subset(singl_idx, labels_all, preds_all, R)
    
    #nonvagueAcc for original singleton examples
    nonvague_acc_singl = acc_subset(singl_idx, true_labels_all, preds_all)
    
    stat_result, prec_recall_f = calculate_metrics(outputs_all, labels_all, R, cutoff, detNN=detNN)

    avg_js_nonvague = stat_result[0] / (stat_result[2]+1e-10)
    avg_js_vague = stat_result[1] / (stat_result[3]+1e-10)
    overall_js = (stat_result[0] + stat_result[1])/(stat_result[2] + stat_result[3]+1e-10)
    js_result = [overall_js, avg_js_vague, avg_js_nonvague]

    test_result_log(
        js_result, prec_recall_f, js_comp, js_singl, 
        nonvague_acc, nonvague_acc_singl, 
        bestModel=bestModel)


def make(args):
    mydata = None
    num_singles = 0
    num_comps = 0
    milestone1 = args.milestone1
    milestone2 = args.milestone2
    device = args.device
    
    if args.dataset == "tinyimagenet":
        mydata = tinyImageNetVague(
            args.data_dir, 
            num_comp=args.num_comp, 
            batch_size=args.batch_size,
            imagenet_hierarchy_path=args.data_dir,
            duplicate=True, #key duplicate
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
            duplicate=True,  #key duplicate
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
            duplicate=True,  #key duplicate
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
            duplicate=True,  #key duplicate
            blur=args.blur,
            gauss_kernel_size=args.gauss_kernel_size,
            pretrain=args.pretrain,
            num_workers=args.num_workers,
            seed=args.seed,
            )
    elif args.dataset == "CIFAR10h":
        mydata = CIFAR10h(
            args.data_dir,
            batch_size=args.batch_size,
            duplicate=True,
            pretrain=args.pretrain,
            num_workers=args.num_workers,
            seed=args.seed,
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
    gauss_kernel_size=args.gauss_kernel_size
    init_lr=args.init_lr
    seed=args.seed
    
    base_path = os.path.join(output_folder, saved_spec_dir)
    tag0 = "_".join([f"{num_comp}M", f"ker{gauss_kernel_size}", f"Seed{seed}", f"BB{args.backbone}", "sweep_DNN"])
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
            valid_loader = mydata.valid_loader
            test_loader = mydata.test_loader
            R = mydata.R
            saved_path = os.path.join(base_path_spec_hyper, "model_CrossEntropy.pt")
            checkpoint = torch.load(saved_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

            model_best_from_valid = copy.deepcopy(model)
            model_best_from_valid.load_state_dict(checkpoint["model_state_dict_best"]) 

            # model after the final epoch
            # bestModel=False
            print(f"\n### Evaluate the model after all epochs:")
            evaluate_vague_nonvague_final(
                model, test_loader, valid_loader, R, num_singles, device, 
                detNN=True, bestModel=False)

            print(f"\n### Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:\n")
            evaluate_vague_nonvague_final(
                model_best_from_valid, test_loader, valid_loader, R, num_singles, device, 
                detNN=True, bestModel=True)


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
