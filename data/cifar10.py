import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from helper_functions import CustomDataset, AddLabelDataset
from data.tinyImageNet import train_valid_split
import time
from collections import defaultdict
from copy import deepcopy
from sklearn.model_selection import train_test_split

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


def extract_file_name_and_labels():
    # file names
    data_path = "/home/cxl173430/data/DATASETS/cifar-10-batches-py/"
    test_data = unpickle(data_path+'test_batch')

    file_name_label_dict = {}
    for idx, (file_name, label_id) in enumerate(zip(test_data[b'filenames'], test_data[b'labels'])):
        file_name = file_name.decode('ascii')
        if file_name not in file_name_label_dict:
            # original without relabeling
            file_name_label_dict[file_name] = label_id
    return file_name_label_dict


def extract_file_name_and_labels_Train():
    # file names
    data_path = "/home/cxl173430/data/DATASETS/cifar-10-batches-py/"
    file_name_label_dict = {}
    for i in range(1,5):
        train_data = unpickle(data_path+'data_batch_'+str(i))
        for idx, (file_name, label_id) in enumerate(zip(train_data[b'filenames'], train_data[b'labels'])):
            file_name = file_name.decode('ascii')
            if file_name not in file_name_label_dict:
                # original without relabeling
                file_name_label_dict[file_name] = label_id
    return file_name_label_dict


class CustomDatasetCIFAR10hTest(Dataset):
    def __init__(self, dataset, file_name_label_dict, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.img_name_label_dict = file_name_label_dict
        
    def __getitem__(self, index):
        x, y = deepcopy(self.dataset[index]) # y: relabeled label, composite label perhaps
        if self.transform:
            x = self.transform(x)

        img_name = self.dataset.imgs[index][0]
        img_name = img_name.split('/')[-1]
        return x, self.img_name_label_dict[img_name], y

    def __len__(self):
        return len(self.dataset)


class CustomDatasetCIFAR10(Dataset):
    def __init__(self, dataset_split, split_idx, dataset, file_name_label_dict, transform=None):
        self.dataset_split = dataset_split
        self.split_idx = split_idx
        self.dataset = dataset
        self.transform = transform
        self.img_name_label_dict = file_name_label_dict
        
    def __getitem__(self, index):
        x, y = deepcopy(self.dataset_split[index]) # y: relabeled label, composite label perhaps
        if self.transform:
            x = self.transform(x)
        # find the original index in the original dataset before splitting
        idx = self.split_idx[index]
        img_name = self.dataset.imgs[idx][0]
        img_name = img_name.split('/')[-1]
        return x, self.img_name_label_dict[img_name], y

    def __len__(self):
        return len(self.dataset_split)


def train_valid_split_local(data, valid_perc=0.1, seed=42):
    # generate indices: instead of the actual data we pass in integers instead
    train_indices, valid_indices, _, _ = train_test_split(
        range(len(data)),
        data.targets,
        stratify=data.targets,
        test_size=valid_perc,
        random_state=seed
    )
    # generate subset based on indices
    train_split = Subset(data, train_indices)
    valid_split = Subset(data, valid_indices)
    return train_split, valid_split, train_indices, valid_indices


class CIFAR10:
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
        print("****** Loading CIFAR10 (relabeled) ... ******")
        start_time = time.time()
        self.batch_size = batch_size
        self.duplicate = duplicate
        self.ratio_train = ratio_train
        self.pretrain = pretrain
        self.num_workers = num_workers
        self.seed = seed
        self.num_classes = 10 # fixed
        
        if not overlap and not overlap_test_only:
            self.data_dir_train = os.path.join(data_dir, 'cifar-10-batches-py/cifar10_train_composites')
            self.data_dir_test = os.path.join(data_dir, 'cifar-10-batches-py/cifar10_test_composites_v2')
            # self.data_dir_test = os.path.join(data_dir, 'cifar-10-batches-py/cifar10_test_composites')
            ALL_LABEL_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck', 
                               'Cat_Dog', 'Deer_Horse', 'Automobile_Truck', 'Airplane_Bird']
            self.vague_classes_ids = [[3,5], [4,7], [1,9], [0,2]]
            # Cat_Dog: [3, 5]
            # Deer_Horse: [4, 7]
            # Automobile_Truck: [1, 9]
            # Airplane_Bird: [0, 2]
            self.R = [[el] for el in range(self.num_classes)]
            for el in self.vague_classes_ids:
                self.R.append(el)
            print(f"Actual label sets\n R: {self.R}")
            self.R_test = self.R

        elif overlap and not overlap_test_only:
            self.data_dir_train = os.path.join(data_dir, 'cifar-10-batches-py/cifar10_train_composites_overlap')
            self.data_dir_test = os.path.join(data_dir, 'cifar-10-batches-py/cifar10_test_composites_overlap')
            ALL_LABEL_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck', 
                               'Cat_Dog', 'Deer_Horse', 'Automobile_Truck', 'Airplane_Bird',
                               'Deer_Dog', 'Deer_Dog_Horse', 'Cat_Deer_Dog', 'Bird_Frog']
            self.vague_classes_ids = [[3,5], [4,7], [1,9], [0,2],
                                      [4,5], [4,5,7], [3,4,5], [2,6]]
            self.R = [[el] for el in range(self.num_classes)]
            for el in self.vague_classes_ids:
                self.R.append(el)
            print(f"Actual label sets\n R: {self.R}")
            self.R_test = self.R
        
        elif overlap_test_only:
            # train: no overlap, test: overlap
            self.data_dir_train = os.path.join(data_dir, 'cifar-10-batches-py/cifar10_train_composites')
            self.data_dir_test = os.path.join(data_dir, 'cifar-10-batches-py/cifar10_test_composites_overlap')
            # self.data_dir_test = os.path.join(data_dir, 'cifar-10-batches-py/cifar10_test_composites')
            ALL_LABEL_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck', 
                               'Cat_Dog', 'Deer_Horse', 'Automobile_Truck', 'Airplane_Bird']
            self.vague_classes_ids = [[3,5], [4,7], [1,9], [0,2]]
            self.vague_classes_ids_test = [[3,5], [4,7], [1,9], [0,2],
                                      [4,5], [4,5,7], [3,4,5], [2,6]]
            self.R_test = [[el] for el in range(self.num_classes)]
            for el in self.vague_classes_ids_test:
                self.R_test.append(el)
            
        self.num_comp = len(self.vague_classes_ids)
        self.kappa = self.num_classes + self.num_comp

        self.img_name_label_dict_Train = extract_file_name_and_labels_Train()
        self.img_name_label_dict = extract_file_name_and_labels()
        
        # Load the ImageFolder dataset
        self.ds_original = datasets.ImageFolder(root=self.data_dir_train)
        self.class_to_idx = self.ds_original.class_to_idx
        self.idx_to_class = {value:key for key, value in self.class_to_idx.items()}
        
        train_split, valid_split, train_idx, valid_idx = train_valid_split_local(self.ds_original, valid_perc=1-ratio_train) #todo: fixed the seed for now
        # train_self.ds_original_n = AddLabelDataset(train_split) #add an aditional label
        # valid_self.ds_original_n = AddLabelDataset(valid_split)
        
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
            norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            pre_norm_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                norm])
            pre_norm_test = transforms.Compose([
                transforms.ToTensor(),
                norm])

        train_ds = CustomDatasetCIFAR10(train_split, train_idx, self.ds_original, self.img_name_label_dict_Train, transform=pre_norm_train)
        valid_ds = CustomDatasetCIFAR10(valid_split, valid_idx, self.ds_original, self.img_name_label_dict_Train, transform=pre_norm_test)
        
        self.ds_test = datasets.ImageFolder(root=self.data_dir_test)
        test_ds = CustomDatasetCIFAR10hTest(self.ds_test, self.img_name_label_dict, transform=pre_norm_test)

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
