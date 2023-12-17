import os
import re
import csv
import json
import numpy as np
import pandas as pd
import random
from PIL import Image
from scipy import io as scio
import math
from math import radians, cos, sin, asin, sqrt, pi

from torchvision import datasets, transforms, utils
from torch.utils.data import sampler, random_split, DataLoader
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import Subset, ConcatDataset
import time
from sklearn.model_selection import train_test_split
from collections import defaultdict

from helper_functions import CustomDataset, AddLabelDataset

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def get_spatial_info(latitude,longitude):
    if latitude and longitude:
        latitude = radians(latitude)
        longitude = radians(longitude)
        x = cos(latitude)*cos(longitude)
        y = cos(latitude)*sin(longitude)
        z = sin(latitude)
        return [x,y,z]
    else:
        return [0,0,0]
def get_temporal_info(date,miss_hour=False):
    try:
        if date:
            if miss_hour:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*)', re.I)
            else:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*) (\d*):(\d*):(\d*)', re.I)
            m = pattern.match(date.strip())

            if m:
                year = int(m.group(1))
                month = int(m.group(2))
                day = int(m.group(3))
                x_month = sin(2*pi*month/12)
                y_month = cos(2*pi*month/12) 
                if miss_hour:
                    x_hour = 0
                    y_hour = 0
                else:
                    hour = int(m.group(4))
                    x_hour = sin(2*pi*hour/24)
                    y_hour = cos(2*pi*hour/24)        
                return [x_month,y_month,x_hour,y_hour]
            else:
                return [0,0,0,0]
        else:
            return [0,0,0,0]
    except:
        return [0,0,0,0]

def find_images_and_targets_nabirds(root,istrain=False):
    root = os.path.join(root,'nabirds')
    image_paths = pd.read_csv(os.path.join(root,'images.txt'),sep=' ',names=['img_id','filepath'])
    image_class_labels = pd.read_csv(os.path.join(root,'image_class_labels.txt'),sep=' ',names=['img_id','target'])
    label_list = list(set(image_class_labels['target']))
    label_list = sorted(label_list)
    label_map = {k: i for i, k in enumerate(label_list)}
    train_test_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])
    data = image_paths.merge(image_class_labels, on='img_id')
    data = data.merge(train_test_split, on='img_id')
    if istrain:
        data = data[data.is_training_img == 1]
    else:
        data = data[data.is_training_img == 0]
    images_and_targets = []
    images_info = []
    for index,row in data.iterrows():
        file_path = os.path.join(os.path.join(root,'images'),row['filepath'])
        target = int(label_map[row['target']])
        images_and_targets.append([file_path,target])
    return images_and_targets,label_map,images_info


def find_images_and_targets(root,istrain=False,aux_info=False):
    if os.path.exists(os.path.join(root,'train.json')):
        with open(os.path.join(root,'train.json'),'r') as f:
            train_class_info = json.load(f)
    elif os.path.exists(os.path.join(root,'train_mini.json')):
        with open(os.path.join(root,'train_mini.json'),'r') as f:
            train_class_info = json.load(f)
    else:
        raise ValueError(f'not eixst file {root}/train.json or {root}/train_mini.json')
    with open(os.path.join(root,'val.json'),'r') as f:
        val_class_info = json.load(f)
    categories_2021 = [x['name'].strip().lower() for x in val_class_info['categories']]
    class_to_idx = {c: idx for idx, c in enumerate(categories_2021)}
    id2label = dict()
    for categorie in train_class_info['categories']:
        id2label[int(categorie['id'])] = categorie['name'].strip().lower()
    class_info = train_class_info if istrain else val_class_info
    
    images_and_targets = []
    images_info = []
    if aux_info:
        temporal_info = []
        spatial_info = []

    for image,annotation in zip(class_info['images'],class_info['annotations']):
        file_path = os.path.join(root,image['file_name'])
        id_name = id2label[int(annotation['category_id'])]
        target = class_to_idx[id_name]
        date = image['date']
        latitude = image['latitude']
        longitude = image['longitude']
        location_uncertainty = image['location_uncertainty']
        images_info.append({'date':date,
                'latitude':latitude,
                'longitude':longitude,
                'location_uncertainty':location_uncertainty,
                'target':target}) 
        if aux_info:
            temporal_info = get_temporal_info(date)
            spatial_info = get_spatial_info(latitude,longitude)
            images_and_targets.append((file_path,target,temporal_info+spatial_info))
        else:
            images_and_targets.append((file_path,target))
    return images_and_targets,class_to_idx,images_info



def get_sample_idx_by_class(dataset, num_classes):
    '''
    Return: [[0,1,...,499],[500, 501,...,999], ..., [..., 99999]]
    200 lists of each 500 samples 
    '''
    sample_idx = []
    sample_idx_by_class = [[] for i in range(num_classes)]
    for i in range(len(dataset)):
        sample_idx_by_class[dataset[i][2]].append(i)
        sample_idx.append(i)
    return sample_idx, sample_idx_by_class


def make_vague_samples(
    dataset, num_single, num_single_comp, vague_classes_ids, 
    blur=True,
    gauss_kernel_size=5, data_train=True):
    trans_blur = None
    if blur:
        sigma_v = 0.3 * ((gauss_kernel_size - 1) * 0.5 - 1) + 0.8
        # sigma_v = gauss_kernel_size / 3
        trans_blur = transforms.GaussianBlur(kernel_size=gauss_kernel_size, sigma=sigma_v)

    all_sample_indices, sample_idx_by_class = get_sample_idx_by_class(dataset, num_single)

    if data_train:
        num_samples_subclass = 45 # 450 for TinyImageNet train
    else:
        num_samples_subclass = 5  # 50 for TinyImageNet valid / test
    
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


class DatasetMeta(Dataset):
    def __init__(
            self,
            root,
            load_bytes=False,
            transform=None,
            train=False,
            aux_info=False,
            dataset='nabirds'
            ):
        self.aux_info = aux_info
        self.dataset = dataset
        if dataset == 'nabirds':
            images,class_to_idx,images_info = find_images_and_targets_nabirds(root,train)

        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.images_info = images_info
        self.load_bytes = load_bytes
        self.transform = transform
        self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index):
        if self.aux_info:
            path, target,aux_info = self.samples[index]
        else:
            path, target = self.samples[index]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.aux_info:
            if type(aux_info) is np.ndarray:
                select_index = np.random.randint(aux_info.shape[0])
                return img, target, aux_info[select_index,:]
            else:
                return img, target, np.asarray(aux_info).astype(np.float64)
        else:
            return img, target

    def __len__(self):
        return len(self.samples)



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


class NabirdsVague():
    def __init__(
        self,
        data_dir,
        batch_size=128,
        ratio_train=0.9,
        duplicate=False,
        blur=True,
        gauss_kernel_size=5,
        pretrain=True,
        num_workers=4,
        seed=42,
        ):
        self.name = "nabirds"
        print('Loading nabirds...')
        start_time = time.time()
        self.root = data_dir
        self.blur = blur
        self.gauss_kernel_size = gauss_kernel_size
        self.duplicate = duplicate
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.num_workers = num_workers
        self.num_classes = 555 #K  
        self.num_comp = 10
        self.kappa = self.num_classes + self.num_comp

        train_ds_original = DatasetMeta(self.root, transform=None,train=True, dataset='nabirds')
        test_ds_original = DatasetMeta(self.root, transform=None,train=False, dataset='nabirds')
        self.class_to_idx = train_ds_original.class_to_idx
        
        self.vague_classes_nids = [['295', '463', '696'],
                                   ['296', '464', '697'],
                                   ['297', '465', '698'],
                                   ['298', '496', '699'],
                                   ['299', '497', '700'],
                                   ['313', '611'],
                                   ['314', '613'],
                                   ['315', '614'],
                                   ['316', '615'],
                                   ['317', '616']]
        self.vague_classes_ids = []
        for el in self.vague_classes_nids:
            self.vague_classes_ids.append([self.class_to_idx[int(k)] for k in el])

        print(f"Vague classes nid: {self.vague_classes_nids}")
        print(f"Vague classes ids: {self.vague_classes_ids}")
        
        self.idx_to_class = {value:key for key, value in self.class_to_idx.items()}
        for i in range(self.num_comp):
            self.idx_to_class[self.num_classes + i] = [self.idx_to_class[k] for k in self.vague_classes_ids[i]]
        # { 0: 'n01443537',
        #   1: 'n01629819',
        # 199: 'n12267677',
        # 200: ['n07871810', 'n07873807', 'n07875152']}
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
            data_train=True)
        valid_ds = make_vague_samples(
            valid_ds_original_n,
            self.num_classes, self.kappa,
            self.vague_classes_ids,
            blur=self.blur,
            gauss_kernel_size=self.gauss_kernel_size,
            data_train=False)
        test_ds = make_vague_samples(
            test_ds_original_n,
            self.num_classes, self.kappa,
            self.vague_classes_ids,
            blur=self.blur,
            gauss_kernel_size=self.gauss_kernel_size,
            data_train=True)
        
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
    root = '/home/cxl173430/data/DATASETS/'
    dataset = 'nabirds'
    dataset = NabirdsVague()
    print(f"class_to_idx: {dataset.class_to_idx}")
    print(f"idx_to_class: {dataset.idx_to_class}")
    
    # images_and_targets,class_to_idx,images_info = find_images_and_targets_nabirds(root, dataset, istrain=False, aux_info=False)

