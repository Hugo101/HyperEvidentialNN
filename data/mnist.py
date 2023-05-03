# to standardize the datasets used in the experiments
# datasets are TinyImageNet, and CIFAR100 (later)
# use create_val_folder() function to convert original Tiny ImageNet structure to structure PyTorch expects

import os
import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import sampler, random_split, DataLoader
from torch.utils.data import Subset, ConcatDataset
from PIL import Image
import random
import math
import time
from sklearn.model_selection import train_test_split
from collections import defaultdict

from helper_functions import CustomDataset, AddLabelDataset


def get_sample_idx_by_class(dataset, num_classes):
    '''
    Train:
    Return: [[5,19,25, ...],[1, 3, 8, ...], ..., [16, 40, 43, ...,]]
    10 lists, each has length: [5331, 6068, 5362, 5518, 5258, 4879, 5326, 5638, 5266, 5354] 
    Valid:
    10 lists, each has length: [592, 674, 596, 613, 584, 542, 592, 627, 585, 595] 
    Test:
    10 lists, each has length: [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
    '''
    sample_idx = []
    sample_idx_by_class = [[] for i in range(num_classes)]
    for i in range(len(dataset)):
        sample_idx_by_class[dataset[i][2]].append(i)
        sample_idx.append(i)
    return sample_idx, sample_idx_by_class


def train_valid_split(data, valid_perc=0.1, seed=42):
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
    return train_split, valid_split


def make_vague_samples(
    dataset, num_single, num_single_comp, vague_classes_ids, 
    blur=True, gray=False,
    gauss_kernel_size=5, data_train=1):
    trans_blur = None
    if blur:
        sigma_v = 0.3 * ((gauss_kernel_size - 1) * 0.5 - 1) + 0.8
        # sigma_v = gauss_kernel_size / 3
        trans_blur = transforms.GaussianBlur(kernel_size=gauss_kernel_size, sigma=sigma_v)
        if gray:
            trans_blur = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                trans_blur,
                ])
    all_sample_indices, sample_idx_by_class = get_sample_idx_by_class(dataset, num_single)

    if data_train==1:
        num_samples_subclass = 14400 # for MNIST train
    elif data_train == 2:
        num_samples_subclass = 1500  # for MNIST valid
    elif data_train == 3:
        num_samples_subclass = 2400  # for MNIST test
    
    total_vague_examples_ids = []
    total_vague_examples = []
    for k in range(num_single, num_single_comp): # i.e.: 200, 201
        # why this num: make sure all classes are balanced
        num_vague = math.floor(num_samples_subclass / (len(vague_classes_ids[k-num_single]) + 1.0)) 
        for subclass in vague_classes_ids[k - num_single]:
            idx_candidates = sample_idx_by_class[subclass]
            vague_ids_subclass = random.sample(idx_candidates, num_vague)
            vague_selected = Subset(dataset, vague_ids_subclass)
            # give new labels for vague images 
            vague_samples = CustomDataset(
                vague_selected, comp_class_id=k, transform=trans_blur)
            total_vague_examples_ids.extend(vague_ids_subclass)
            total_vague_examples.append(vague_samples)
            
    idx_non_candidates = list(set(all_sample_indices)-set(total_vague_examples_ids))
    nonvague_examples = Subset(dataset, idx_non_candidates)
    
    for vague_exam in total_vague_examples:
        nonvague_examples += vague_exam
    
    return nonvague_examples



class MNIST():
    def __init__(
        self, 
        data_dir, 
        num_comp=4, 
        batch_size=128, 
        ratio_train=0.9, 
        duplicate=False,
        blur=True,
        gray=False,
        gauss_kernel_size=5,
        pretrain=False, #if using pretrained model, resize img to 224
        num_workers=4, #tune this on different servers
        seed=42):
        self.name = "mnist"
        print('Loading mnist...')
        start_time = time.time()
        self.blur = blur
        self.gray = gray
        self.gauss_kernel_size = gauss_kernel_size
        self.duplicate = duplicate
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.num_workers = num_workers
        # self.img_size = 32 #28
        self.num_classes = 10 #K  
        self.num_comp = num_comp
        self.kappa = self.num_classes + self.num_comp

        # original MNIST
        train_ds_original = datasets.MNIST(data_dir, train=True, download=True)
        test_ds_original = datasets.MNIST(data_dir, train=False, download=True)
        self.class_to_idx = train_ds_original.class_to_idx
        # classes_to_idx
        # {'0 - zero': 0,
        # '1 - one': 1,
        # '2 - two': 2,
        # '3 - three': 3,
        # '4 - four': 4,
        # '5 - five': 5,
        # '6 - six': 6,
        # '7 - seven': 7,
        # '8 - eight': 8,
        # '9 - nine': 9}

        self.parent_to_subclasses = {
            10: [3, 8],
            11: [4, 7],
            12: [1, 9],
            13: [2, 5]} # 10, 11, 12, 13 are the vague classes
        self.candidate_superclasses = list(self.parent_to_subclasses.keys())
        self.candidate_superclasses = sorted(self.candidate_superclasses)
        print(f"Total {len(self.candidate_superclasses)} Candidate superclasses: {self.candidate_superclasses}")
        self.vague_classes_ids = [self.parent_to_subclasses[k] for k in self.candidate_superclasses]
        # [[3, 8], [4, 7], [1, 9], [2, 5]]
        print(f"Vague classes ids: {self.vague_classes_ids}")
        
        self.idx_to_class = {value:key for key, value in self.class_to_idx.items()}
        for i in range(self.num_comp):
            self.idx_to_class[self.num_classes + i] = [self.idx_to_class[k] for k in self.vague_classes_ids[i]]
        # { 0: '0 - zero', 1: '1 - one', ..., 
        #  10: ['3 - three', '8 - eight']}
        self.R = [[el] for el in range(self.num_classes)] 
        for el in self.vague_classes_ids:
            self.R.append(el)
        print(f"Actual label sets\n R: {self.R}")

        train_split, valid_split = train_valid_split(train_ds_original, valid_perc=1-ratio_train, seed=seed)
        train_ds_original_n = AddLabelDataset(train_split) #add an aditional label
        valid_ds_original_n = AddLabelDataset(valid_split)
        test_ds_original_n = AddLabelDataset(test_ds_original)

        train_ds = make_vague_samples(
            train_ds_original_n,
            self.num_classes, self.kappa,
            self.vague_classes_ids,
            blur=self.blur,
            gray=self.gray,
            gauss_kernel_size=self.gauss_kernel_size,
            data_train=1)
        valid_ds = make_vague_samples(
            valid_ds_original_n,
            self.num_classes, self.kappa,
            self.vague_classes_ids,
            blur=self.blur,
            gray=self.gray,
            gauss_kernel_size=self.gauss_kernel_size,
            data_train=2)
        test_ds = make_vague_samples(
            test_ds_original_n,
            self.num_classes, self.kappa,
            self.vague_classes_ids,
            blur=self.blur,
            gray=self.gray,
            gauss_kernel_size=self.gauss_kernel_size,
            data_train=3)
        
        if self.pretrain:
            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        train_ds = CustomDataset(train_ds, transform=pre_norm_train)
        valid_ds = CustomDataset(valid_ds, transform=pre_norm_test)
        test_ds = CustomDataset(test_ds, transform=pre_norm_test)

        if self.duplicate:
            train_ds = self.modify_vague_samples(train_ds)
            valid_ds = self.modify_vague_samples(valid_ds)
        
        print(f'Data splitted. Train, Valid, Test size: {len(train_ds), len(valid_ds), len(test_ds)}')
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        
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
        
        subset_1 = Subset(dataset, idx1)  # the rest 
        
        for comp_label, indx in idx2.items(): # each vague example
            comp_label_subset = Subset(dataset, indx)
            copies = CustomDataset(comp_label_subset, comp_class_id=C[comp_label - self.num_classes][0])
            for j in range(1, len(C[comp_label - self.num_classes])):
                copies += CustomDataset(comp_label_subset, comp_class_id=C[comp_label - self.num_classes][j])
            subset_1 = subset_1 + copies
        return subset_1


if __name__ == '__main__':
    data_dir = '/home/cxl173430/data/DATASETS/'
    batch_size = 64
    dataset = MNIST(
            data_dir,
            batch_size=batch_size,
            duplicate=True)

    print(f"class_to_idx: {dataset.class_to_idx}")
    print(f"idx_to_class: {dataset.idx_to_class}")
    print(f"parent_to_subclasses: {dataset.parent_to_subclasses}")
    print(f"candicate superclasses: {dataset.candidate_superclasses}")