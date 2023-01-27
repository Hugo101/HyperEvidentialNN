import numpy as np
import torch
import pickle as pkl
import os
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from HyperEvidentialNN.robustness.robustness.tools import folder
from HyperEvidentialNN.robustness.robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26
from HyperEvidentialNN.robustness.robustness.tools.helpers import get_label_mapping



def get_training_dataloader_breeds(ds_name, info_dir, data_dir, batch_size=16, num_workers=2, shuffle=True):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ])

    breeds_training = BREEDS(
        info_dir=info_dir,
        data_dir=data_dir,
        ds_name=ds_name,
        partition='train',
        split=None,
        transform=transform_train,
        train=True,
        seed=1000,
        tr_ratio=0.9)
    breeds_training_loader = DataLoader(breeds_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=False)

    return breeds_training_loader


def get_validation_dataloader_breeds(ds_name, info_dir, data_dir, batch_size=16, num_workers=2, shuffle=True):
    transform_validation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ])

    breeds_validation = BREEDS(
        info_dir=info_dir,
        data_dir=data_dir,
        ds_name=ds_name,
        partition='train',
        split=None,
        transform=transform_validation,
        train=False,
        seed=1000,
        tr_ratio=0.9)
    breeds_validation_loader = DataLoader(breeds_validation, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=False)

    return breeds_validation_loader


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
            images, labels, coarse_labels = shuffle(images, labels, coarse_labels, random_state=seed)
            num_tr = int(len(images) * tr_ratio)

            if train is True:
                self.images = images[:num_tr]
                self.labels = labels[:num_tr]
                self.coarse_labels = coarse_labels[:num_tr]
            else:
                self.images = images[num_tr:]
                self.labels = labels[num_tr:]
                self.coarse_labels = coarse_labels[num_tr:]
        else:
            self.images = images
            self.labels = labels
            self.coarse_labels = coarse_labels


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, coarse_target = self.images[index], self.labels[index], self.coarse_labels[index]

        if self.transform is not None:
            img = self.transform(self.loader(img))

        return img, target, coarse_target, index

    def __len__(self):
        return len(self.images)

