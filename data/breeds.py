import os
import sys
path_test = os.path.abspath(os.path.join(os.getcwd()))
print("#### ", path_test)
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
import time
import math
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from helper_functions import CustomDataset
from HyperEvidentialNN.robustness.robustness.tools import folder
from HyperEvidentialNN.robustness.robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26
from HyperEvidentialNN.robustness.robustness.tools.helpers import get_label_mapping


class BREEDSFactory:
    def __init__(self, info_dir, data_dir):
        self.info_dir = info_dir
        self.data_dir = data_dir


    def get_breeds(self, ds_name, partition, transforms=None, split=None):
        superclasses, subclass_split, label_map = self.get_classes(ds_name, split)
        partition = 'val' if partition == 'validation' else partition
        print(f"==> Preparing BREEDS dataset {ds_name}, partition: {partition}..")
        return self.create_dataset(partition, subclass_split[0], transforms)


    def create_dataset(self, partition, subclass_split, transforms):
        coarse_custom_label_mapping = get_label_mapping("custom_imagenet", subclass_split)
        fine_subclass_split = [[item] for sublist in subclass_split for item in sublist]
        fine_custom_label_mapping = get_label_mapping("custom_imagenet", fine_subclass_split)

        active_custom_label_mapping = fine_custom_label_mapping
        active_subclass_split = fine_subclass_split

        dataset = folder.ImageFolder(root=os.path.join(self.data_dir, partition), transform=transforms, label_mapping=active_custom_label_mapping)
        coarse2fine, coarse_labels = self.extract_c2f_from_dataset(dataset, coarse_custom_label_mapping, fine_custom_label_mapping, partition)
        setattr(dataset, 'num_classes', len(active_subclass_split))
        setattr(dataset, 'coarse2fine', coarse2fine)
        setattr(dataset, 'coarse_targets', coarse_labels)
        return dataset


    def extract_c2f_from_dataset(self, dataset, coarse_custom_label_mapping, fine_custom_label_mapping, partition):
        classes, original_classes_to_idx = dataset._find_classes(os.path.join(self.data_dir, partition))
        _, coarse_classes_to_idx = coarse_custom_label_mapping(classes, original_classes_to_idx)
        _, fine_classes_to_idx = fine_custom_label_mapping(classes, original_classes_to_idx)
        coarse2fine = {}
        for k, v in coarse_classes_to_idx.items():
            if v in coarse2fine:
                coarse2fine[v].append(fine_classes_to_idx[k])
            else:
                coarse2fine[v] = [fine_classes_to_idx[k]]

        # modification
        # ---
        fine2coarse = {}
        for k in coarse2fine:
            fine_labels_k = coarse2fine[k]
            for i in range(len(fine_labels_k)):
                assert fine_labels_k[i] not in fine2coarse
                fine2coarse[fine_labels_k[i]] = k

        fine_labels = dataset.targets
        coarse_labels = []
        for i in range(len(fine_labels)):
            coarse_labels.append(fine2coarse[fine_labels[i]])

        return coarse2fine, coarse_labels


    def get_classes(self, ds_name, split=None):
        if ds_name == 'living17':
            return make_living17(self.info_dir, split)
        elif ds_name == 'entity30':
            return make_entity30(self.info_dir, split)
        elif ds_name == 'entity13':
            return make_entity13(self.info_dir, split)
        elif ds_name == 'nonliving26':
            return make_nonliving26(self.info_dir, split)
        else:
            raise NotImplementedError


class BREEDS(Dataset):
    def __init__(
        self, info_dir, data_dir, ds_name, partition, split, transform, 
        train=True, seed=1000, tr_ratio=0.9):
        super(Dataset, self).__init__()
        breeds_factory = BREEDSFactory(info_dir, data_dir)
        self.dataset = breeds_factory.get_breeds(
            ds_name=ds_name,
            partition=partition,
            transforms=None,
            split=split)

        self.transform = transform
        self.loader = self.dataset.loader

        images = [s[0] for s in self.dataset.samples]
        labels = self.dataset.targets
        coarse_labels = self.dataset.coarse_targets

        if partition == 'train':
            train_indices, valid_indices, _, _ = train_test_split(
                range(len(self.dataset)),
                self.dataset.targets,
                stratify=self.dataset.targets,
                test_size=1-tr_ratio,
                random_state=seed
            )

            if train is True:
                self.images = [images[i] for i in train_indices]
                self.labels = [labels[i] for i in train_indices]
                self.coarse_labels = [coarse_labels[i] for i in train_indices]
            else:
                self.images = [images[i] for i in valid_indices]
                self.labels = [labels[i] for i in valid_indices]
                self.coarse_labels = [coarse_labels[i] for i in valid_indices]
        else:
            self.images = images
            self.labels = labels
            self.coarse_labels = coarse_labels


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, coarse_target) where target is index of the target class.
        """
        img, target, coarse_target = self.images[index], self.labels[index], self.coarse_labels[index]

        if self.transform is not None:
            img = self.transform(self.loader(img))

        return img, target, target # here we need to return target, target for convenience

    def __len__(self):
        return len(self.images)


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
    
    # print()
    print(f"A: {num_single}, {num_single_comp}, {vague_classes_ids}")
    all_sample_indices, sample_idx_by_class = get_sample_idx_by_class(dataset, num_single)
    print("B: ")
    total_vague_examples_ids = []
    total_vague_examples = []
    for k in range(num_single, num_single_comp):
        print("C: ")
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
    print("D")
    for vague_exam in total_vague_examples:
        nonvague_examples += vague_exam
    
    return nonvague_examples


class BREEDSVague:
    def __init__(
        self,
        info_dir, 
        data_dir, 
        ds_name,
        num_comp=1,
        batch_size=128,
        tr_ratio=0.9,
        duplicate=False,
        blur=True,
        gauss_kernel_size=3,
        pretrain=True,
        num_workers=4,
        seed=1000,
        comp_el_size=2):
        
        self.name = f"BREEDS-{ds_name}"
        print(f"Loading {self.name} ...")
        start_time = time.time()
        
        self.blur = blur
        self.gauss_kernel_size = gauss_kernel_size
        self.duplicate = duplicate
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.num_workers = num_workers
        self.comp_el_size = comp_el_size
        self.ratio_train = tr_ratio
        
        transform_pre = transforms.Compose([
        transforms.Resize(224)
        ])
        
        breeds_training = BREEDS(
            info_dir=info_dir,
            data_dir=data_dir,
            ds_name=ds_name,
            partition='train', #key
            split=None,
            transform=transform_pre,
            train=True,  #key
            seed=seed,
            tr_ratio=self.ratio_train)
        
        self.num_classes = breeds_training.dataset.num_classes
        self.num_comp = num_comp
        self.kappa = self.num_classes + self.num_comp
        
        breeds_validation = BREEDS(
            info_dir=info_dir,
            data_dir=data_dir,
            ds_name=ds_name,
            partition='train',
            split=None,
            transform=transform_pre,
            train=False,
            seed=seed,
            tr_ratio=self.ratio_train)
        
        breeds_test = BREEDS(
            info_dir=info_dir,
            data_dir=data_dir,
            ds_name=ds_name,
            partition='val',
            split=None,
            transform=transform_pre,
            train=False)
        
        # Get hierachical representation
        self.parent_to_subclasses = breeds_training.dataset.coarse2fine
        print(f"Hierarchical labels: {self.parent_to_subclasses}\n")
        self.candidate_superclasses = list(self.parent_to_subclasses.keys())
        print(f"Total {len(self.candidate_superclasses)} Candidate superclasses: {self.candidate_superclasses}")
        self.vague_classes_nids, self.vague_classes_ids = self.get_vague_classes_breeds() 
        print(f"Vague classes nid: {self.vague_classes_nids}")
        print(f"Vague classes ids: {self.vague_classes_ids}")
        
        self.R = [[el] for el in range(self.num_classes)] 
        for el in self.vague_classes_ids:
            self.R.append(el)
        print(f"Actual label sets\n R: {self.R}")
        
        train_ds = make_vague_samples(
            breeds_training, 
            self.num_classes, self.kappa,
            self.vague_classes_ids, 
            blur=self.blur,
            gauss_kernel_size=self.gauss_kernel_size,
            num_samples_subclass=1170) # num of samples per class for train
        valid_ds = make_vague_samples(
            breeds_validation, 
            self.num_classes, self.kappa, 
            self.vague_classes_ids,
            blur=self.blur,
            gauss_kernel_size=self.gauss_kernel_size,
            num_samples_subclass=130) # num of samples per class for valid
        test_ds = make_vague_samples(
            breeds_test, 
            self.num_classes, self.kappa, 
            self.vague_classes_ids,
            blur=self.blur,
            gauss_kernel_size=self.gauss_kernel_size,
            num_samples_subclass=50) # num of samples per class for test
        
        if self.pretrain:
            norm=transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm])
            transform_test = transforms.Compose([
                transforms.Resize(256),     # Resize images to 256 x 256
                transforms.CenterCrop(224), # Center crop image
                transforms.ToTensor(),
                norm])
        else:
            raise NotImplementedError

        train_ds = CustomDataset(train_ds, transform=transform_train)
        valid_ds = CustomDataset(valid_ds, transform=transform_test)
        test_ds = CustomDataset(test_ds, transform=transform_test)
        
        print(f'## Final (w/o duplicate) Train, Valid, Test size: {len(train_ds), len(valid_ds), len(test_ds)}')
        
        if self.duplicate:
            train_ds = self.modify_vague_samples(train_ds)
            valid_ds = self.modify_vague_samples(valid_ds)
            print(f'## Final Train, Valid, Test size: {len(train_ds), len(valid_ds), len(test_ds)}')

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)
        
        time_load = time.time() - start_time
        print(f"Loading data finished. Time: {time_load//60:.0f}m {time_load%60:.0f}s")


    def get_vague_classes_breeds(self):
        vague_classes = random.sample(self.candidate_superclasses, self.num_comp)
        
        vague_subs_nids = [random.sample(self.parent_to_subclasses[super], self.comp_el_size) for super in vague_classes]

        vague_subs_ids = vague_subs_nids

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
