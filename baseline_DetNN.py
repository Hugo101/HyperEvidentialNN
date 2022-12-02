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
args = opt

# config = {
#     "dataset": "tinyimagenet",
#     "backbone": "EfficientNet-b3",
#     "epochs": 1,
#     "init_lr": 0.0001,
#     "train": True,
#     "test": True,
#     "nonVaguePred": True,
#     "vaguePred": True,
# }

# base_path = "HyperEvidentialNN/models_baseline_DetNN"
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
            running_loss_GT = 0.0
            running_corrects_GT = 0.0
            epoch_acc, epoch_acc_GT = 0.0, 0.0
            # Iterate over data.
            for batch_idx, (inputs, single_labels_GT, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                single_labels_GT = single_labels_GT.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    else:
                        loss = criterion(outputs, single_labels_GT)

                # statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects_GT += torch.sum(preds == single_labels_GT.data)

                if phase == "train":
                    running_corrects += torch.sum(preds == labels.data)

            if scheduler is not None:
                if phase == "train":
                    scheduler.step()

            # print(f"##### length of datasets at phase {phase}: {len(dataloader.dataset)}") #pass 
            epoch_loss = running_loss / len(dataloader.dataset)
            if phase == "train":
                epoch_acc = running_corrects / len(dataloader.dataset)
                epoch_acc = epoch_acc.cpu().item()

            epoch_acc_GT = running_corrects_GT / len(dataloader.dataset)
            epoch_acc_GT = epoch_acc_GT.cpu().item()
            
            train_log(phase, epoch, epoch_acc, epoch_acc_GT, epoch_loss)

            if phase == "train":
                time_epoch_train = time.time() - begin_epoch
                print(
                f"Finish the Train in this epoch in {time_epoch_train//60:.0f}m {time_epoch_train%60:.0f}s.")

            if phase == "val" and epoch_acc_GT > best_acc:
                best_acc = epoch_acc_GT
                best_epoch = epoch
                print(f"The best epoch: {best_epoch}, acc: {best_acc:.4f}.")
                best_model_wts = copy.deepcopy(model.state_dict()) # deep copy the model
                # torch.save(state, f'{logdir}/tiny_{epoch}_{best_acc:.4f}.pt')
            # if phase == "val":
            #     if epoch == 0 or ((epoch+1) % 1 ==0):
            #         acc = evaluate(model, mydata.test_loader, mydata.valid_loader, epoch, device=device)
            #         state = {
            #                 "model_state_dict": model.state_dict(),
            #             }
            #         torch.save(state, f'{logdir}/tiny_{epoch}_{acc:.4f}.pt')

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
def evaluate_nonvague_final(
    model,
    test_loader,
    device
    ):
    model.eval()
    total_correct = 0.0
    total_samples = 0
    # losses = []
    for batch in test_loader:
        images, single_labels_GT, labels = batch
        images, labels = images.to(device), labels.to(device)
        single_labels_GT = single_labels_GT.to(device)
        output = model(images)
        # loss = criterion(output, labels)
        _, preds = torch.max(output, 1)
        total_correct += torch.sum(preds == single_labels_GT.data) # nonvague prediction
        total_samples += len(labels)
        # val_loss = loss.detach()
        # val_losses.append(val_loss)
    # loss = torch.stack(val_losses).mean().item()
    acc = total_correct / total_samples
    return acc


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
        images, labels = images.to(device), labels.to(device)
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
    end_cutoff = 1
    interval = 0.05
    accs = []
    cutoffs = []
    while cutoff <= end_cutoff:
        accuracy = evaluate_cutoff(model, val_loader, R, cutoff, device, detNN=detNN)
        print(f"For cutoff = {cutoff:.2f}, Validation Accuracy: {accuracy:.4f}")    
        accs.append(accuracy)
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
def evaluate_vague_final(
    model, 
    test_loader, 
    val_loader, 
    R, 
    device,
    detNN=True):
    cutoff = get_cutoff(model, val_loader, R, device, detNN=detNN)
    print(f"### selected cutoff: {cutoff}")
    model.eval()
    outputs_all = []
    labels_all = []
    preds_all = []
    for batch in test_loader:
        images, _, labels = batch
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        preds = output.argmax(dim=1)

        outputs_all.append(output)
        labels_all.append(labels)
        preds_all.append(preds)

    outputs_all = torch.cat(outputs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    stat_result, prec_recall_f = calculate_metrics(outputs_all, labels_all, R, cutoff, detNN=detNN)

    avg_js_nonvague = stat_result[0] / (stat_result[2]+1e-10)
    avg_js_vague = stat_result[1] / (stat_result[3]+1e-10)
    overall_js = (stat_result[0] + stat_result[1])/(stat_result[2] + stat_result[3]+1e-10)
    js_result = [overall_js, avg_js_vague, avg_js_nonvague]

    return js_result, prec_recall_f


def test_result_log(js_result, prec_recall_f, acc, bestModel=False):
    if bestModel:
        wandb.log({
            f"TestB OverallJS": js_result[0], 
            f"TestB CompJS": js_result[1], 
            f"TestB SnglJS": js_result[2],
            f"TestB CompPreci": prec_recall_f[0], 
            f"TestB CompRecal": prec_recall_f[1], 
            f"TestB CompFscor": prec_recall_f[2], 
            f"TestB acc": acc})
        print(f"TestBest accNonVague: {acc:.4f}, \n \
            JS(O_V_N): {js_result}, P_R_F_compGTcnt_cmpPREDcnt: {prec_recall_f}\n")
    else:
        wandb.log({
            f"TestF OverallJS": js_result[0], 
            f"TestF CompJS": js_result[1], 
            f"TestF SnglJS": js_result[2],
            f"TestF CompPreci": prec_recall_f[0], 
            f"TestF CompRecal": prec_recall_f[1], 
            f"TestF CompFscor": prec_recall_f[2], 
            f"TestF acc": acc})
        print(f"TestF accNonVague: {acc:.4f}, \n \
            JS(O_V_N): {js_result}, P_R_F_compGTcnt_cmpPREDcnt: {prec_recall_f}\n")


def make(args):
    mydata = None
    num_singles = 0
    num_comps = 0

    if args.dataset == "tinyimagenet":
        mydata = tinyImageNetVague(
            args.data_dir, 
            num_comp=args.num_comp, 
            batch_size=args.batch_size,
            imagenet_hierarchy_path=args.data_dir,
            duplicate=True) #key duplicate 
        num_singles = mydata.num_classes
        num_comps = mydata.num_comp
        print(f"Data: {args.dataset}, num of singleton and composite classes: {num_singles, num_comps}")
    # elif args.dataset == "cifar100":
        # dataset = cifar100Vague()

    if args.backbone == "EfficientNet-b3":
        model = EfficientNet_pretrain(num_singles)
    else:
        print(f"### ERROR: The backbone {args.backbone} is invalid!")
    model = model.to(device)
    print("### Loss type: CrossEntropy (no uncertainty)")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    return mydata, model, criterion, optimizer, scheduler


def main():
    print('Random Seed: {}'.format(args.seed))
    set_random_seeds(args.seed)

    mydata, model, criterion, optimizer, scheduler = make(args)

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
        saved_path = os.path.join(base_path, "model_CrossEntropy.pt")
        torch.save(state, saved_path)
        print(f"Saved: {saved_path}")
        end = time.time()
        print(f'Total training time for ENN: {(end-start)//60:.0f}m {(end-start)%60:.0f}s')
    
    if args.test:
        valid_loader = mydata.valid_loader
        test_loader = mydata.test_loader
        R = mydata.R
        saved_path = os.path.join(base_path, "model_CrossEntropy.pt")
        checkpoint = torch.load(saved_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        model_best_from_valid = copy.deepcopy(model)
        model_best_from_valid.load_state_dict(checkpoint["model_state_dict_best"]) 

        # model after the final epoch
        if args.vaguePred:
            js_result, prec_recall_f = evaluate_vague_final(model, test_loader, valid_loader, R, device)
        if args.nonVaguePred:
            acc_nonvague = evaluate_nonvague_final(model, test_loader, device)
        test_result_log(js_result, prec_recall_f, acc_nonvague, False) # bestModel=False

        print(f"### Use the model selected from validation set in Epoch {checkpoint['epoch_best']}:\n")
        if args.vaguePred:
            js_result, prec_recall_f = evaluate_vague_final(model_best_from_valid, test_loader, valid_loader, R, device)
        if args.nonVaguePred:
            acc_nonvague = evaluate_nonvague_final(model_best_from_valid, test_loader, device)
        test_result_log(js_result, prec_recall_f, acc_nonvague, True)


if __name__ == "__main__":
    # tell wandb to get started
    print(config)
    with wandb.init(project=f"ENN-Vague-{config['dataset']}-DetNN", config=config):
        config = wandb.config
        main()