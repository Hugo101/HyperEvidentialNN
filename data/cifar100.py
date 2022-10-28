import torch
import os
from torchvision import datasets, transforms, utils
from torch.utils.data import sampler, random_split, DataLoader, ConcatDataset
from PIL import Image
import random

class CIFAR100:
    def __init__(self, root_dir, batch_size=128):
        self.name = "cifar100"
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 100
        self.num_test = 10000
        # self.num_train = 50000
        ratio_train = 0.9
        self.num_train = int(50000 * ratio_train)
        self.num_val = 50000 - self.num_train

        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        self.augmented = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), normalize])
        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        # augment
        self.aug_trainset = datasets.CIFAR100(root=root_dir, train=True, download=True, transform=self.augmented)
        # self.aug_train_loader = DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        aug_train_ds, aug_val_ds = random_split(self.aug_trainset, [self.num_train, self.num_val],
                                                generator=torch.Generator().manual_seed(42))
        self.aug_train_loader = DataLoader(aug_train_ds, batch_size=batch_size, shuffle=True,
                                           num_workers=4)
        self.aug_valid_loader = DataLoader(aug_val_ds, batch_size=batch_size, shuffle=False,
                                           num_workers=4)

        # no augment
        self.trainset = datasets.CIFAR100(root=root_dir, train=True, download=True, transform=self.normalized)
        # self.train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        train_ds, val_ds = random_split(self.trainset, [self.num_train, self.num_val],
                                        generator=torch.Generator().manual_seed(42))
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                       num_workers=4)
        self.train_loader_sf_false = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                       num_workers=4)
        self.valid_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                       num_workers=4)

        # Test
        self.testset = datasets.CIFAR100(root=root_dir, train=False, download=True, transform=self.normalized)
        self.test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False,
                                      num_workers=4)
