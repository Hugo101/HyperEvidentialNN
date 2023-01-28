import torch
import os
from torchvision import datasets, transforms, utils
from torch.utils.data import sampler, random_split, DataLoader, ConcatDataset, Subset
from PIL import Image
import random
import math
from helper_functions import CustomDataset, AddLabelDataset
from data.tinyImageNet import train_valid_split
import tarfile
import time
from torchvision.datasets.utils import download_url

def create_path(dir):
    if dir is not None:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    print(dir)


class CIFAR100:
    def __init__(self, data_dir, batch_size=128):
        self.name = "cifar100"
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 100
        self.num_test = 10000
        # self.num_train = 50000
        ratio_train = 0.9
        self.num_train = int(50000 * ratio_train)
        self.num_val = 50000 - self.num_train

        normalize = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], 
            std=[0.267, 0.256, 0.276])
        self.augmented = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(32, padding=4), 
            transforms.ToTensor(), 
            normalize
            ]
                                            )
        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        # augment
        self.aug_trainset = datasets.CIFAR100(
            root=data_dir, train=True, download=True, 
            transform=self.augmented)
        # self.aug_train_loader = DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        aug_train_ds, aug_val_ds = random_split(
            self.aug_trainset, 
            [self.num_train, self.num_val],
            generator=torch.Generator().manual_seed(42))
        self.aug_train_loader = DataLoader(
            aug_train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4)
        self.aug_valid_loader = DataLoader(
            aug_val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4)

        # no augment
        self.trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=self.normalized)
        # self.train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        train_ds, val_ds = random_split(
            self.trainset, 
            [self.num_train, self.num_val],
            generator=torch.Generator().manual_seed(42))
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4)
        self.train_loader_sf_false = DataLoader(
            train_ds, batch_size=batch_size, shuffle=False,
            num_workers=4)
        self.valid_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4)

        # Test
        self.testset = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=self.normalized)
        self.test_loader = DataLoader(
            self.testset, batch_size=batch_size, shuffle=False,
            num_workers=4)


def get_sample_idx_by_class(dataset, num_classes):
    sample_idx = []
    sample_idx_by_class = [[] for i in range(num_classes)]
    for i in range(len(dataset)):
        sample_idx_by_class[dataset[i][2]].append(i)
        sample_idx.append(i)
    return sample_idx, sample_idx_by_class


def make_vague_samples(
    dataset, num_single, num_single_comp, vague_classes_ids, 
    blur=True, 
    gauss_kernel_size=3, num_samples_subclass=450):
    trans_blur = None
    if blur:
        sigma_v = 0.3 * ((gauss_kernel_size - 1) * 0.5 - 1) + 0.8
        # sigma_v = gauss_kernel_size / 3
        trans_blur = transforms.GaussianBlur(kernel_size=gauss_kernel_size, sigma=sigma_v)
    all_sample_indices, sample_idx_by_class = get_sample_idx_by_class(dataset, num_single)

    total_vague_examples_ids = []
    total_vague_examples = []
    for k in range(num_single, num_single_comp):
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


def get_ParentToSubclasses(hierarchy_dir):
    data_dir = hierarchy_dir + 'data/cifar100'
    hierarchical_labels = {}
    superclasses = []
    for superClass in sorted(os.listdir(data_dir + "/train")):
        superclasses.append(superClass)
        hierarchical_labels[superClass] = sorted(os.listdir(data_dir + "/train/" + superClass))
    print("Hierarchical labels:\n")
    for superClass in hierarchical_labels:
        print(superClass, ': ', hierarchical_labels[superClass])
    return hierarchical_labels


class CIFAR100Vague:
    def __init__(
        self, 
        data_dir,
        num_comp=1, 
        batch_size=128,
        ratio_train=0.9,
        duplicate=False,
        hierarchy_dir="/home/cxl173430/data/DATASETS/cifar100_henn/",
        blur=True,
        gauss_kernel_size=3,
        pretrain=True,
        num_workers=4,
        seed=42,
        comp_el_size=2):
        self.name = "cifar100"
        print("Loading CIFAR100...")
        start_time = time.time()
        self.blur = blur
        self.gauss_kernel_size = gauss_kernel_size
        self.duplicate = duplicate
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.num_workers = num_workers
        self.comp_el_size = comp_el_size
        self.img_size = 32
        self.num_classes = 100
        self.num_comp = num_comp
        self.kappa = self.num_classes + self.num_comp
        num_train = 50000
        self.ratio_train = ratio_train
        self.num_train = int(num_train * self.ratio_train)
        self.num_val = num_train - self.num_train
        self.num_test = 10000

        train_ds_original = datasets.CIFAR100(root=data_dir, download=True)
        test_ds_original = datasets.CIFAR100(root=data_dir, train=False)
        self.class_to_idx = train_ds_original.class_to_idx

        #get hiretical
        create_path(hierarchy_dir)
        cifar100zipfile = hierarchy_dir+'cifar100.tgz'
        if not os.path.exists(cifar100zipfile):
            dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz"
            download_url(dataset_url, hierarchy_dir)
            # Extract from archive
            with tarfile.open(cifar100zipfile, 'r:gz') as tar:
                tar.extractall(path=hierarchy_dir+'data')

        self.parent_to_subclasses = get_ParentToSubclasses(hierarchy_dir)
        self.candidate_superclasses = list(self.parent_to_subclasses.keys())
        print(f"Total {len(self.candidate_superclasses)} Candidate superclasses: {self.candidate_superclasses}")
        self.vague_classes_nids, self.vague_classes_ids = self.get_vague_classes_v2() 
        print(f"Vague classes nid: {self.vague_classes_nids}")
        print(f"Vague classes ids: {self.vague_classes_ids}")
        
        self.idx_to_class = {value:key for key, value in self.class_to_idx.items()}
        for i in range(self.num_comp):
            self.idx_to_class[self.num_classes + i] = [self.idx_to_class[k] for k in self.vague_classes_ids[i]]
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
            gauss_kernel_size=self.gauss_kernel_size,
            num_samples_subclass=450) # num of samples per class for train
        valid_ds = make_vague_samples(
            valid_ds_original_n, 
            self.num_classes, self.kappa, 
            self.vague_classes_ids,
            blur=self.blur,
            gauss_kernel_size=self.gauss_kernel_size,
            num_samples_subclass=50) # num of samples per class for valid
        test_ds = make_vague_samples(
            test_ds_original_n, 
            self.num_classes, self.kappa, 
            self.vague_classes_ids,
            blur=self.blur,
            gauss_kernel_size=self.gauss_kernel_size,
            num_samples_subclass=100) # num of samples per class for test

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

        train_ds = CustomDataset(train_ds, transform=pre_norm_train)
        valid_ds = CustomDataset(valid_ds, transform=pre_norm_test)
        test_ds = CustomDataset(test_ds, transform=pre_norm_test)

        if self.duplicate:
            train_ds = self.modify_vague_samples(train_ds)
            valid_ds = self.modify_vague_samples(valid_ds)
        
        print(f'Data splitted. Train, Valid, Test size: {len(train_ds), len(valid_ds), len(test_ds)}')
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)
        
        time_load = time.time() - start_time
        print(f"Loading data finished. Time: {time_load//60:.0f}m {time_load%60:.0f}s")


    def get_vague_classes_v2(self):
        vague_classes = random.sample(self.candidate_superclasses, self.num_comp)
        # vague_classes = ["people"]
        
        vague_subs_nids = [random.sample(self.parent_to_subclasses[super], self.comp_el_size) for super in vague_classes]
        # vague_subs_nids = [['baby', 'woman']]
        vague_subs_ids = [[self.class_to_idx[sub] for sub in super] for super in vague_subs_nids]

        #   C = [[self.class_to_idx[sub] for sub in random.sample(self.parent_to_subclasses[super_class], comp_el_size)] for super_class in vague_classes]

        print(f'Vague classes: {vague_classes}')
        return vague_subs_nids, vague_subs_ids
    
    
    def modify_vague_samples(self, dataset):
        C = self.vague_classes_ids
        for k in range(self.num_classes, self.kappa): # K, kappa
            # idx1 = [i for i in range(len(dataset)) if dataset[i][2] != k]
            # idx2 = [i for i in range(len(dataset)) if dataset[i][2] == k] 
            idx1 = []
            idx2 = []
            for i in range(len(dataset)):
                if dataset[i][2] != k:
                    idx1.append(i)
                else:
                    idx2.append(i)

            subset_1 = Subset(dataset, idx1)  # the rest 
            subset_2 = Subset(dataset, idx2)  #vague composite k
            copies = CustomDataset(subset_2, comp_class_id=C[k - self.num_classes][0])
            for j in range(1, len(C[k - self.num_classes])):
                copies += CustomDataset(subset_2, comp_class_id=C[k - self.num_classes][j])
            dataset = subset_1 + copies
        return dataset