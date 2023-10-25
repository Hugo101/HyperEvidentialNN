import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
data_path = "/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/"
sys.path.insert(1, data_path)
import torch
import numpy as np
import time
import yaml
import wandb 
import torch
torch.set_num_threads(4)
import torch.nn as nn
from config_args import parser  
from common_tools import create_path, set_device, dictToObj, set_random_seeds
from data.tinyImageNet import tinyImageNetVague
from data.cifar100 import CIFAR100Vague
from data.breeds import BREEDSVague
from data.mnist import MNIST
from backbones import HENN_EfficientNet, HENN_ResNet50, HENN_VGG16, HENN_LeNet, HENN_LeNet_v2

from hiclass import LocalClassifierPerNode
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make(args):
    mydata = None
    num_singles = 0
    num_comps = 0
    num_classes_both = 0 
    
    ### Dataset ###
    if args.dataset == "tinyimagenet":
        mydata = tinyImageNetVague(
            args.data_dir, 
            num_comp=args.num_comp, 
            batch_size=args.batch_size,
            imagenet_hierarchy_path=args.data_dir,
            blur=args.blur,
            gray=False,
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
    if args.dataset == "mnist":
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
    
    ### Backbone ###
    num_classes_both = num_singles + num_comps
    if args.backbone == "EfficientNet-b3":
        model = HENN_EfficientNet(num_classes_both, pretrain=args.pretrain)
    elif args.backbone == "ResNet50":
        model = HENN_ResNet50(num_classes_both)
    elif args.backbone == "VGG16":
        model = HENN_VGG16(num_classes_both)
    elif args.backbone == "LeNet":
        model = HENN_LeNet(num_classes_both)
    elif args.backbone == "LeNetV2":
        model = HENN_LeNet_v2(out_dim=num_classes_both)
    else:
        print(f"### ERROR {args.dataset}: The backbone {args.backbone} is invalid!")
    model = model.to(args.device)

    return mydata, model


def generate_label(mydata, split='train'):
    if split == 'train':
        data_tuple = mydata.train_loader.dataset
    elif split == 'test':
        data_tuple = mydata.test_loader.dataset
    
    label_GT = []
    label_comp = []
    num_data = len(data_tuple)
    for i in range(num_data):
        label_GT.append(data_tuple[i][1])
        label_comp.append(data_tuple[i][2])

    ### labels
    labels = []
    for i in range(num_data):
        if label_GT[i] != label_comp[i]:
            tmp = [label_comp[i], '']
        else:
            tmp = [label_comp[i], label_GT[i]]
            # tmp = ['rest', train_label_GT[i]]
        labels.append(tmp)
    return  labels


def extract_features(mydata, split, model_path, pretrained_model, device):
    # load pretrained model
    saved_path_HENN = os.path.join(model_path, "model_uncertainty_gdd.pt")
    checkpoint = torch.load(saved_path_HENN, map_location=device)
    pretrained_model.load_state_dict(checkpoint["model_state_dict_best"])
    pretrained_model.eval()
    # extract features
    avgPool = nn.AdaptiveAvgPool2d(1)
    features_all = []
    if split == "train":
        data_loader = mydata.train_loader
    elif split == "test":
        data_loader = mydata.test_loader
    with torch.no_grad():
        for batch_idx, (inputs, targets_GT, labels) in enumerate(data_loader):
            inputs = inputs.to(device, non_blocking=True)
            features_batch = pretrained_model.network.extract_features(inputs)
            features_batch = avgPool(features_batch)
            features_batch = features_batch.reshape(features_batch.shape[0], -1)
            features_all.append(features_batch)
    
    data_features = torch.cat(features_all, dim=0)
    
    return data_features


def generate_data_labels(mydata, split, model_path, pretrained_model, device):
    data_features = extract_features(mydata, split, model_path, pretrained_model, device)
    labels = generate_label(mydata, split=split)
    return data_features.cpu().numpy(), labels


def calculate_metrics(predictions, labels, num_singles, R):
    correct_vague = 0.0
    correct_nonvague = 0.0
    vague_total = 0
    nonvague_total = 0
    num_corr = 0
    for i in range(len(predictions)):
        # singleton acc
        if predictions[i][1]: # singleton prediction
            if int(predictions[i][1]) == labels[i][1]:
                num_corr += 1

        # composite 
        if predictions[i][0]: # composite prediction
            predicted_set = set(R[int(predictions[i][0])])
            k = labels[i][0] # ground truth
            ground_truth_set = set(R[k])
            intersect = predicted_set.intersection(ground_truth_set)
            union = predicted_set.union(ground_truth_set)
            rate = len(intersect) / len(union)
            if int(predictions[i][0])>=num_singles:
                correct_vague += rate
                vague_total += 1
            else:
                correct_nonvague += rate
                nonvague_total += 1
    stat_result = [correct_nonvague, correct_vague, nonvague_total,vague_total]
    print("statistics: ", stat_result)
    
    overall_js = (stat_result[0] + stat_result[1])/(len(predictions)+1e-10)
    avg_js_vague = stat_result[1] / (stat_result[3]+1e-10)
    acc = num_corr / len(predictions)
    
    return overall_js, avg_js_vague, acc


def evaluate_vague_result_log(
    overJS, compJS, SinglAcc
    ):
    tag = "Test"
    wandb.log({
        f"{tag} JSoverall": overJS, 
        f"{tag} JScomp": compJS,
        f"{tag} SinglAcc": SinglAcc
        })
    print(f"{tag} SinglAcc: {SinglAcc:.4f},\n\
        JS(O_V_N): {overJS:.4f}, {compJS:.4f},\n")

# class EfficientNetFeat(nn.Module):
#     def __init__(self, pretrained_model):
#         super(EfficientNetFeat, self).__init__()
#         self.features = nn.Sequential(
#             *list(pretrained_model.network.children())[:-2]
#         )
#     def forward(self, x):
#         x = self.features(x)
#         return x


def main(args):
    print(f"Current all hyperparameters: {args}")
    set_random_seeds(args.seed)
    mydata, model = make(args)
    num_singles = mydata.num_classes
    num_classes = num_singles + mydata.num_comp
    print("Total number of classes to train: ", num_classes)
    pretrained_model_path = "/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN_Results/Tiny/GDD_20_15_10M_ker357_sweep_entrGDD_0906/SEED42_10M_Ker5_sweep_GDDexp101/lr_1e-05_klLam_0.0_EntrLamDir_0.0_EntrLamGDD_0"
    device = args.device
    train_data, train_labels = generate_data_labels(mydata, 'train', pretrained_model_path, model, device)
    test_data, test_labels = generate_data_labels(mydata, 'test', pretrained_model_path, model, device)
    print("Training data shape: ", train_data.shape)
    print("Test data shape: ", test_data.shape)
    
    X_train = train_data
    Y_train = train_labels

    X_test = test_data
    # Use random forest classifiers for every node
    rf = RandomForestClassifier()
    # rf = LogisticRegression()
    classifier = LocalClassifierPerNode(local_classifier=rf)

    # Train local classifier per node
    classifier.fit(X_train, Y_train)

    # Predict
    predictions = classifier.predict(X_test)
    print(predictions)
    overJS, compJS, SinglAcc = calculate_metrics(predictions, test_labels, mydata.num_classes, mydata.R)
    evaluate_vague_result_log(overJS, compJS, SinglAcc)


if __name__ == "__main__":
    args = parser.parse_args()
    opt = vars(args)

    # build the path to save model and results
    create_path(args.output_folder) 
    base_path = os.path.join(args.output_folder, args.saved_spec_dir)
    create_path(base_path)

    config_file = os.path.join(base_path, "config_Hier.yml")
    CONFIG = yaml.load(open(config_file), Loader=yaml.FullLoader)
    opt.update(CONFIG)
    opt["device"] = set_device(args.gpu)

    # tell wandb to get started
    print("Default setting before hyperparameters tuning:", opt)
    project_name = f"{opt['dataset']}-{opt['num_comp']}M-Ker{opt['gauss_kernel_size']}-HierarchCls-Debug"
    with wandb.init(project=project_name, config=opt):
        config = wandb.config
        main(config)
