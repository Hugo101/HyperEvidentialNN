import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from helper_functions import CustomDataset, AddLabelDataset
from data.tinyImageNet import train_valid_split
import time
from collections import defaultdict
from copy import deepcopy
from sklearn.model_selection import train_test_split
import pickle 

def create_path(dir):
    if dir is not None:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    print(dir)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CustomDatasetFmnistTest(Dataset):
    def __init__(self, dataset, pred_sets_list, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.pred_sets_list = pred_sets_list
        
    def __getitem__(self, index):
        x, y = deepcopy(self.dataset[index]) # y: singleton Ground truth label
        if self.transform:
            x = self.transform(x)
        # find the pred set (singleton or composite label) for this example
        y_comp = self.pred_sets_list[index] # 0, 1, ..., 11, 12 ...
        return x, y, y_comp

    def __len__(self):
        return len(self.dataset)


class CustomDatasetFmnist(Dataset):
    def __init__(self, dataset_split, split_idx, pred_sets_list, transform=None):
        self.dataset_split = dataset_split
        self.split_idx = split_idx
        self.transform = transform
        self.pred_sets_list = pred_sets_list
        
    def __getitem__(self, index):
        x, y = deepcopy(self.dataset_split[index]) # y: singleton Ground truth label
        if self.transform:
            x = self.transform(x)
        # find the original index in the original dataset before splitting
        idx = self.split_idx[index]
        # find the predicted set (singleton or composite label) for this example
        y_comp = self.pred_sets_list[idx] # 0, 1, ..., 11, 12 ...
        return x, y, y_comp

    def __len__(self):
        return len(self.dataset_split)


def train_valid_split_local(data, pred_targets, valid_perc=0.1, seed=42):
    # generate indices: instead of the actual data we pass in integers instead
    train_indices, valid_indices, _, _ = train_test_split(
        range(len(data)),
        pred_targets,
        stratify=pred_targets,
        test_size=valid_perc,
        random_state=seed
    )
    # generate subset based on indices
    train_split = Subset(data, train_indices)
    valid_split = Subset(data, valid_indices)
    return train_split, valid_split, train_indices, valid_indices


def string_2_int(pred_set_train_selected):
    string_2_int_dict = {
        "0_6": 10,
        "2_4": 11,
        "7_9": 12,
        "2_6": 13,
        "4_6": 14,
        "3_4": 15,
        "0_2": 16,
        "3_6": 17,
        "0_3": 18,
        "5_7": 19
    }
    pred_set_train_selected_int = []
    for string_label in pred_set_train_selected:
        str_list = string_label.split("_")
        if len(str_list) == 1:
            # singleton label
            pred_set_train_selected_int.append(int(str_list[0]))
        else:
            # composite label
            if string_label in string_2_int_dict:
                pred_set_train_selected_int.append(string_2_int_dict[string_label])
            else:
                print("Error: string_2_int_dict does not contain this string label")
                
    return pred_set_train_selected_int


class FMNIST:
    def __init__(
        self,
        data_dir,
        batch_size=128,
        duplicate=False,
        ratio_train=0.9,
        pretrain=True,
        num_workers=4,
        seed=42,
        overlap=False,
        overlap_test_only=False
    ):
        print("****** Loading Fashion MNIST (synthetic composite labels) ... ******")
        start_time = time.time()
        self.batch_size = batch_size
        self.duplicate = duplicate
        self.ratio_train = ratio_train
        self.pretrain = pretrain
        self.num_workers = num_workers
        self.seed = seed
        self.num_classes = 10 # fixed
        saved_idx_dir = "/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/data/saved_models/Fmnist_saved_idx_Categories_Top20.pkl"
        saved_idx_dir_nonoverlap = "/home/cxl173430/data/uncertainty_Related/HENN_Git_VScode/HyperEvidentialNN/data/saved_models/Fmnist_saved_idx_Categories_Top13.pkl"
        if not overlap and not overlap_test_only:
            # train: no overlap, test: no overlap
            self.vague_classes_ids = [[0,6], [2,4], [7,9]]
            # T-shirt/top_Shirt: [0, 6]
            # Pullover_Coat: [2, 4]
            # Sneaker_AnkleBoot: [7, 9]
            self.R = [[el] for el in range(self.num_classes)]
            for el in self.vague_classes_ids:
                self.R.append(el)
            print(f"Actual label sets\n R: {self.R}")
            self.R_test = self.R

        elif overlap and not overlap_test_only:
            # train: overlap, test: overlap
            self.vague_classes_ids = [[0,6], [2,4], [7,9],
                                      [2,6], [4,6], [3,4],[0,2],[3,6],[0,3],[5,7]]
            self.R = [[el] for el in range(self.num_classes)]
            for el in self.vague_classes_ids:
                self.R.append(el)
            print(f"Actual label sets\n R: {self.R}")
            self.R_test = self.R
        
        elif overlap_test_only:
            # train: no overlap, test: overlap
            self.vague_classes_ids = [[0,6], [2,4], [7,9]]
            self.vague_classes_ids_test = [[0,6], [2,4], [7,9],
                                           [2,6], [4,6], [3,4],[0,2],[3,6],[0,3],[5,7]]
            self.R = [[el] for el in range(self.num_classes)]
            for el in self.vague_classes_ids:
                self.R.append(el)
            self.R_test = [[el] for el in range(self.num_classes)]
            for el in self.vague_classes_ids_test:
                self.R_test.append(el)
            
        self.num_comp = len(self.vague_classes_ids)
        self.kappa = self.num_classes + self.num_comp
        self.kappa_test = len(self.R_test)
        
        if self.pretrain:
            norm = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            pre_norm_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm])
            pre_norm_test = transforms.Compose([
                transforms.Resize(256),     # Resize images to 256 x 256
                transforms.CenterCrop(224), # Center crop image
                transforms.ToTensor(),
                norm])
        else:
            norm = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            pre_norm_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                norm])
            pre_norm_test = transforms.Compose([
                transforms.ToTensor(),
                norm])

        # Load original Fashion MNIST
        self.train_ds_original = datasets.FashionMNIST(data_dir, train=True, download=True, transform=pre_norm_train)
        self.test_ds_original = datasets.FashionMNIST(data_dir, train=False, download=True, transform=pre_norm_test)
        self.class_to_idx = self.train_ds_original.class_to_idx
        self.idx_to_class = {value:key for key, value in self.class_to_idx.items()}
        
        # Load the saved info
        if not overlap and not overlap_test_only:
            # train: no overlap, test: no overlap
            saved_idx = pickle.load(open(saved_idx_dir_nonoverlap, "rb"))
            sample_idx_train = saved_idx[0]
            sample_idx_test = saved_idx[1]
            pred_set_train_selected = saved_idx[2] # "0", "0_2"
            pred_set_test_selected = saved_idx[3]
        elif overlap and not overlap_test_only:
            # train: overlap, test: overlap
            saved_idx = pickle.load(open(saved_idx_dir, "rb"))
            sample_idx_train = saved_idx[0]
            sample_idx_test = saved_idx[1]
            pred_set_train_selected = saved_idx[4] # "0", "0_2"
            pred_set_test_selected = saved_idx[5]
        elif overlap_test_only:
            # train: no overlap, test: overlap
            saved_idx_train = pickle.load(open(saved_idx_dir_nonoverlap, "rb"))
            sample_idx_train = saved_idx_train[0]
            pred_set_train_selected = saved_idx_train[2]
            
            saved_idx_test = pickle.load(open(saved_idx_dir, "rb"))
            sample_idx_test = saved_idx_test[1]
            pred_set_test_selected = saved_idx_test[5]

        self.train_ds_selected = Subset(self.train_ds_original, sample_idx_train)
        self.test_ds_selected = Subset(self.test_ds_original, sample_idx_test)

        train_split, valid_split, train_idx, valid_idx = train_valid_split_local(self.train_ds_selected, pred_set_train_selected, valid_perc=1-ratio_train) #todo: fixed the seed for now
        
        # convert the string label to int
        # "0" -> 0,  "0_6"-> 10
        pred_set_train_selected_int = string_2_int(pred_set_train_selected)
        assert max(pred_set_train_selected_int) == self.kappa - 1
        pred_set_test_selected_int = string_2_int(pred_set_test_selected)
        assert max(pred_set_test_selected_int) == self.kappa_test - 1        
        train_ds = CustomDatasetFmnist(train_split, train_idx, pred_set_train_selected_int)
        # print("~~~~~~~~~~~ two labels:", train_ds[1][1], train_ds[1][2])
        valid_ds = CustomDatasetFmnist(valid_split, valid_idx, pred_set_train_selected_int)
        test_ds = CustomDatasetFmnistTest(self.test_ds_selected, pred_set_test_selected_int)

        if self.duplicate:
            train_ds = self.modify_vague_samples(train_ds)
            valid_ds = self.modify_vague_samples(valid_ds)

        print(f'Data splitted. Train, Valid, Test size: {len(train_ds), len(valid_ds), len(test_ds)}')
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)
        self.test_loader  = DataLoader(test_ds,  batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)
        time_load = time.time() - start_time
        
        print(f"Loading data finished. Time: {time_load//60:.0f}m {time_load%60:.0f}s")
        
        
    def modify_vague_samples(self, dataset):
        C = self.vague_classes_ids
        idx1 = [] # singleton example idx
        idx2 = defaultdict(list) # vague examples label and their idx
        for i in range(len(dataset)):
            # print("~~~~~~~~~~~ type of dataset[i]", type(dataset))
            # print("~~~~~~~~~~~ dataset", dataset)
            if dataset[i][2] >= self.num_classes:
                # composite example
                idx2[dataset[i][2]].append(i)
            else:
                idx1.append(i)
        
        subset_1 = Subset(dataset, idx1)  # singleton examples 
        
        for comp_label, indx in idx2.items(): # each vague class
            comp_label_subset = Subset(dataset, indx)
            copies = CustomDataset(comp_label_subset, comp_class_id=C[comp_label - self.num_classes][0])
            for j in range(1, len(C[comp_label - self.num_classes])):
                copies += CustomDataset(comp_label_subset, comp_class_id=C[comp_label - self.num_classes][j])
            subset_1 = subset_1 + copies
        return subset_1
