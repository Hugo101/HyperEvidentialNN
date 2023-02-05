import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import copy
import time
import yaml
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

import wandb
from collections import Counter
from random import randint

import torch
torch.set_num_threads(4)

import random
from scipy.optimize import minimize
import math
import numpy as np

from torch import optim, nn

# DS related 
import dst_pytorch as ds_layer
from dst_pytorch import EfficientNet_DS, DM_set_test

data_path = "/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/"
sys.path.insert(1, data_path)
from backbones import EfficientNet_pretrain, ResNet50
from data.tinyImageNet import tinyImageNetVague
from data.cifar100 import CIFAR100Vague
from common_tools import set_device, create_path, dictToObj, set_random_seeds

parser = argparse.ArgumentParser(description='Conformalize Torchvision Model')
parser.add_argument('--data_dir', default="/home/cxl173430/data/DATASETS/", type=str, help='path to dataset')
parser.add_argument(
    "--output_folder", 
    default="/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results/", 
    type=str, help="where results will be saved."
)
parser.add_argument(
    "--saved_spec_dir", default="CIFAR100/10M_ker3_ECNN_pytorchKer_V2", 
    type=str, help="specific experiment path."
    )
parser.add_argument('--gpu', default=0, type=int, help='GPU ID')
parser.add_argument('--seed', default=42, type=int, help='random seed')

args = parser.parse_args()
opt = vars(args)
# build the path to save model and results
create_path(args.output_folder) 
base_path = os.path.join(args.output_folder, args.saved_spec_dir)
create_path(base_path)
config_file = os.path.join(base_path, "config_ECNN.yml")
config = yaml.load(open(config_file), Loader=yaml.FullLoader)
opt.update(config)
args = opt
args = dictToObj(args)

device = set_device(args.gpu)


## Helper Functions
# aim func: cross entropy
def func(x):
    fun=0
    for i in range(len(x)):
        fun += x[i] * math.log10(x[i])
    return fun

#constraint 1: the sum of weights is 1
def cons1(x):
    return sum(x)

#constraint 2: define tolerance to imprecision
def cons2(x):
    tol = 0
    for i in range(len(x)):
        tol += (len(x) -(i+1)) * x[i] / (len(x) - 1)
    return tol


#function for power set
def PowerSetsBinary(items):  
    #generate all combination of N items  
    N = len(items)  
    #enumerate the 2**N possible combinations  
    set_all=[]
    for i in range(2**N):
        combo = []  
        for j in range(N):  
            if(i >> j ) % 2 == 1:  
                combo.append(items[j]) 
        set_all.append(combo)
    return set_all


def average_utility(utility_matrix, inputs, labels, act_set):
    utility = 0
    for i in range(len(inputs)):
        x = inputs[i]
        y = labels[i]
        utility += utility_matrix[x,y]
    average_utility = utility/len(inputs)
    return average_utility
### Helper Functions END


def generateUtilityMatrix(num_class, act_set):
    #compute the weights g for ordered weighted average aggreagtion
    for j in range(2,(num_class+1)):
        if j > 3:
            break
        print(f"\n############# j: {j}")
        num_weights = j
        ini_weights = np.asarray(np.random.rand(num_weights))
        print(f"initial weights: {ini_weights}")
        
        name='weight'+str(j)
        locals()['weight'+str(j)]= np.zeros([5, j])

        for i in range(5):
            print(f"       ^^^^^^^^^^^ i: {i}")
            tol = 0.5 + i * 0.1

            cons = ({'type': 'eq', 'fun' : lambda x: cons1(x)-1},
                {'type': 'eq', 'fun' : lambda x: cons2(x)-tol},
                {'type': 'ineq', 'fun' : lambda x: x-0.00000001}
                )

            res = minimize(func, ini_weights, method='SLSQP', options={'disp': True}, constraints=cons)
            locals()['weight'+str(j)][i] = res.x
            print (res.x)
    
    utility_matrix = np.zeros([len(act_set), num_class])
    tol_i = 3 
    #tol_i = 0 with tol=0.5, tol_i = 1 with tol=0.6, tol_i = 2 with tol=0.7, tol_i = 3 with tol=0.8, tol_i = 4 with tol=0.9
    for i in range(len(act_set)):
        intersec = act_set[i]
        if len(intersec) == 1:
            utility_matrix[i, intersec] = 1
    
        else:
            for j in range(len(intersec)):
                utility_matrix[i, intersec[j]] = locals()['weight'+str(len(intersec))][tol_i, 0]
    # print(utility_matrix)
    return utility_matrix


def train_valid_log(phase, epoch, accDup, accGT, loss):
    wandb.log({
        f"{phase} epoch": epoch, 
        f"{phase} loss": loss, 
        f"{phase} accDup": accDup, 
        f"{phase} accGT": accGT}, step=epoch)
    print(f"{phase.capitalize()} loss: {loss:.4f} accDup: {accDup:.4f} accGT: {accGT:.4f}")


def test_result_log_ECNN(
    js_result, 
    prec_recall_f, 
    accs, 
    bestModel=True):
    if bestModel:
        tag = "TestB"
    else:
        tag = "TestF"
    
    wandb.log({
        f"{tag}_JSoverall": js_result[0], 
        f"{tag}_JScomp": js_result[1], 
        f"{tag}_JSsngl": js_result[2],
        f"{tag}_CmpPreci": prec_recall_f[0], 
        f"{tag}_CmpRecal": prec_recall_f[1], 
        f"{tag}_CmpFscor": prec_recall_f[2], 
        f"{tag}_NonVagueAcc0": accs,
        }, step=None)
    print(f"{tag}_NonVagueAcc: {accs:.4f}, \n \
            JS(O_V_N): {js_result[0]:.4f}, {js_result[1]:.4f}, {js_result[2]:.4f}, \n \
            P_R_F_compGTcnt_cmpPREDcnt: {prec_recall_f}\n")


def make(args, device):
    mydata = None
    num_singles = 0
    milestone1 = args.milestone1
    milestone2 = args.milestone2
    # device = args.device
    if args.dataset == "tinyimagenet":
        mydata = tinyImageNetVague(
                    args.data_dir, 
                    num_comp=args.num_comp, 
                    batch_size=args.batch_size,
                    imagenet_hierarchy_path=args.data_dir,
                    duplicate=True,
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
                    duplicate=True,
                    blur=args.blur,
                    gauss_kernel_size=args.gauss_kernel_size,
                    pretrain=args.pretrain,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    comp_el_size=args.num_subclasses,
                    )
        
    print(f"Size of training/validation/test:")
    print(len(mydata.train_loader.dataset)) # >90200 because of the duplicates
    print(len(mydata.valid_loader.dataset))
    print(len(mydata.test_loader.dataset))

    num_singles = mydata.num_classes
    num_comps = mydata.num_comp
    print(f"Data: {args.dataset}, num of singleton and composite classes: {num_singles, num_comps}")

    # define pretrained CNN model
    num_singles = mydata.num_classes
    if args.backbone == "EfficientNet-b3":
        model_prob = EfficientNet_pretrain(num_singles)
    elif args.backbone == "ResNet50":
        model_prob = ResNet50(num_singles)
    model_prob = model_prob.to(device)

    # define DS layer
    if args.backbone == "EfficientNet-b3":
        model_DS = EfficientNet_DS(num_singles)
    elif args.backbone == "ResNet50":
        model_DS = ResNet50(num_singles) # todo:need to add resnet modification for DS
    model_DS = model_DS.to(device)
    
    # define model_DS and update related parameters using model_prob parameters
    # load pretrained model_prob (traditional DNN)
    
    saved_spec_dir_DNN = args.saved_spec_dir_DNN
    model_saved_base_path = os.path.join(args.output_folder, saved_spec_dir_DNN)
    print("DNN model saved path:", model_saved_base_path)
    saved_path = os.path.join(model_saved_base_path, "model_CrossEntropy.pt")
    # load pretrained CNN model
    checkpoint = torch.load(saved_path, map_location=device)
    model_prob.load_state_dict(checkpoint["model_state_dict_best"]) 
    
    pretrained_dict = model_prob.state_dict()
    model_DS_dict = model_DS.state_dict()

    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_DS_dict}
    model_DS_dict.update(pretrained_dict)
    model_DS.load_state_dict(model_DS_dict)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model_DS.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone1, milestone2], gamma=0.1)
    
    return mydata, model_DS, criterion, optimizer, scheduler


def train_ds_model(
    model,
    mydata,
    criterion,
    optimizer,
    scheduler=None,
    num_epoch=25,
    device=None,
):
    wandb.watch(model, log="all", log_freq=100)
    since = time.time()
    dataloader = mydata.train_loader
    dataset_size_train = len(dataloader.dataset)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0.0

    criterion = nn.NLLLoss() # todo: Check this further
    
    for epoch in range(num_epoch):
        begin_epoch = time.time()
        print(f"Epoch {epoch}/{num_epoch-1}")
        print("-"*10)
        # Each epoch has a training and validation phase
        print("Training...")
        print(f"get last lr:{scheduler.get_last_lr()}") if scheduler else ""

        model.train() # set model to training mode
        running_loss = 0.0
        running_corrects = 0.0
        running_corrects_GT = 0
        
        # Iterate over data
        for batch_idx, (inputs, single_label_GT, labels) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            single_labels_GT = single_labels_GT.to(device, non_blocking=True)
            
            #zero the parameter gradients
            optimizer.zero_grad()
            #forward
            outputs = model(inputs) #shape: batch*(classes+1)
            
            #utility layer 
            logits = ds_layer.DM(model.n_classes, 0.9, device=device)(outputs) # todo
            loss = criterion(torch.log(logits+1e-8), labels)
            _, preds = torch.max(logits, 1)

            loss.backward()
            optimizer.step()
            
            # statistics
            batch_size = inputs.size(0)
            running_loss += loss.detach() * batch_size
            running_corrects += torch.sum(preds == labels)
            running_corrects_GT += torch.sum(preds == single_labels_GT)
        
        if scheduler:
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

        #validation phase
        valid_acc, valid_loss = validate_ds_model(
            model, mydata.valid_loader, criterion, device)
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


def validate_ds_model(model, dataloader, criterion, device):
    print("Validating...")
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    dataset_size_val = len(dataloader.dataset)
    for batch_idx, (inputs, single_labels_GT, single_label_dup) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        labels = single_label_dup.to(device, non_blocking=True)
        # forward
        with torch.no_grad():
            outputs = model(inputs)
            #utility layer 
            logits = ds_layer.DM(model.n_classes, 0.9, device=device)(outputs) # todo
            loss = criterion(torch.log(logits+1e-8), labels)
            _, preds = torch.max(logits, 1)
        
        # statistics
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels)
    
    epoch_loss = running_loss / dataset_size_val
    epoch_acc = running_corrects / dataset_size_val
    epoch_acc = epoch_acc.detach()
    return epoch_acc, epoch_loss


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


def calculate_metrics_set_prediction(outputs, labels, act_set, num_single_class):
    correct_vague = 0.0
    correct_nonvague = 0.0
    vague_total = 0
    nonvague_total = 0
    predSet_or_not =[]
    
    # Get the predicted labels
    pred_label_ids = torch.argmax(outputs, dim=1)

    for i in range(len(labels)):
        act_idx_local = pred_label_ids[i]
        predicted_set = set(act_set[act_idx_local])        
        if len(predicted_set) == 1:
            predSet_or_not.append(0) # singleton
        else:
            predSet_or_not.append(1)

        ground_truth_set = set(act_set[labels[i].item()])
        inter_set = predicted_set.intersection(ground_truth_set)
        union_set = predicted_set.union(ground_truth_set)
        rate = float(len(inter_set)/len(union_set))
        
        if len(predicted_set) == 1:
            correct_nonvague += rate
            nonvague_total += 1
        else:
            correct_vague += rate 
            vague_total += 1
    
    stat_result = [correct_nonvague, correct_vague, nonvague_total, vague_total]
    
    predSet_or_not = torch.tensor(predSet_or_not) #1:vague, 0：non-vague
    # check precision, recall, f-score for composite classes
    prec_r_f = precision_recall_f_v1(labels, predSet_or_not, num_single_class)
    return stat_result, prec_r_f


@torch.no_grad()
def evaluate_ds_set_prediction(model, test_loader, num_class, act_set, nu, utility_matrix):
    num_set = len(act_set)
    dmTest = DM_set_test(num_class, num_set, nu).to(device)  
    dmTest.weight = nn.Parameter(torch.tensor(utility_matrix).T)
    
    model.eval()
    outputs_all = []
    labels_all = []
    
    for batch in test_loader:
        images, single_labels_GT, labels = batch
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        # single_labels_GT = single_labels_GT.to(device, non_blocking=True)
        ds_normalize = model(images)
        output = dmTest(ds_normalize)
        outputs_all.append(output)
        labels_all.append(labels)
    
    outputs_all = torch.cat(outputs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    stat_result, prec_recall_f = calculate_metrics_set_prediction(outputs_all, labels_all, act_set, num_class)
    return stat_result, prec_recall_f


def evaluate_vague_nonvague_final_ECNN(
    model,
    test_loader,
    R,
    num_singles,
    device,
    nu = 0.9,
    bestModel=False
):
    print("### SingletonPrediction (nonVagueAcc):")
    
    # begin evaluation
    model.eval()
    labels_all = []
    true_labels_all = []
    preds_all = []
    total_correct = 0.0
    total_samples = 0
    for batch in test_loader:
        images, single_labels_GT, labels = batch
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        single_labels_GT = single_labels_GT.to(device, non_blocking=True)
        outputs = model(images)
        #utility layer 
        logits = ds_layer.DM(model.n_classes, 0.9, device=device)(outputs) # todo
        _, preds = torch.max(logits, 1)

        total_correct += torch.sum(preds == single_labels_GT) # nonvague
        total_samples += len(labels)
        
        labels_all.append(labels)
        preds_all.append(preds)
        true_labels_all.append(single_labels_GT)
    print(f"Total samples in test set: {total_samples}")
    # nonvague prediction accuracy
    nonvague_acc = total_correct / total_samples  

    print("### Set Prediction: ")
    print("Generate utility matrix:")
    utility_matrix = generateUtilityMatrix(num_singles, R)
    stat_result, prec_recall_f = evaluate_ds_set_prediction(model, test_loader, num_singles, R, nu, utility_matrix)

    avg_js_nonvague = stat_result[0] / (stat_result[2]+1e-10)
    avg_js_vague = stat_result[1] / (stat_result[3]+1e-10)
    overall_js = (stat_result[0] + stat_result[1])/(stat_result[2] + stat_result[3]+1e-10)
    js_result = [overall_js, avg_js_vague, avg_js_nonvague]
    
    test_result_log_ECNN(
        js_result, prec_recall_f,
        nonvague_acc, 
        bestModel=bestModel)


def generateSpecPath(
    output_folder, saved_spec_dir, 
    num_comp,
    gauss_kernel_size,
    init_lr):
    base_path = os.path.join(output_folder, saved_spec_dir)
    tag0 = "_".join([f"{num_comp}M", f"ker{gauss_kernel_size}", "sweep_E-CNN"])
    base_path_spec_hyper_0 = os.path.join(base_path, tag0)
    create_path(base_path_spec_hyper_0)
    base_path_spec_hyper = os.path.join(base_path_spec_hyper_0, str(init_lr))
    create_path(base_path_spec_hyper)
    return base_path_spec_hyper


def main(args):
    base_path_spec_hyper = generateSpecPath(
            args.output_folder, args.saved_spec_dir, 
            args.num_comp,
            args.gauss_kernel_size,
            args.init_lr)
    
    ## Fix randomness
    set_random_seeds(seed=args.seed) # 42
    device = set_device(args.gpu)
    mydata, model, criterion, optimizer, scheduler = make(args, device)
    num_singles = mydata.num_classes
    
    if args.train:
        start = time.time()
        model, model_best, epoch_best = train_ds_model(
            model, 
            mydata, 
            criterion,
            optimizer,
            scheduler=scheduler,
            num_epoch=args.epochs,
            device=device
            )
        state = {
            "epoch_best": epoch_best,
            "model_state_dict": model.state_dict(),
            "model_state_dict_best": model_best.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            }
    
        saved_path = os.path.join(base_path_spec_hyper, "model_uncertainty_ECNN.pt")
        torch.save(state, saved_path)
        print(f"Saved: {saved_path}")
        end = time.time()
        print(f'Total training time for ENN: {(end-start)//60:.0f}m {(end-start)%60:.0f}s')
    else:
        print(f"## No training, load trained model directly")

    if args.test:
        test_loader = mydata.test_loader
        R = mydata.R
        
        saved_path = os.path.join(base_path_spec_hyper, "model_uncertainty_ECNN.pt")
        checkpoint = torch.load(saved_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        model_best_from_valid = copy.deepcopy(model)
        model_best_from_valid.load_state_dict(checkpoint["model_state_dict_best"]) 

        # model after the final epoch
        print(f"\n### Evaluate the model after all epochs:")
        evaluate_vague_nonvague_final_ECNN(
                model, test_loader, R, num_singles, device, 
                nu=0.9, bestModel=False)

        print(f"\n### Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:\n")
        evaluate_vague_nonvague_final_ECNN(
                model_best_from_valid, test_loader, R, num_singles, device,
                nu=0.9, bestModel=True)


if __name__ == "__main__":
    # tell wandb to get started
    print("A:", config)
    with wandb.init(project=f"{config['dataset']}-{config['num_comp']}M-RAPS", config=args):
        config = wandb.config
        print("B:", config)
        main(config)