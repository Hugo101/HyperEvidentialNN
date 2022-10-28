# to standardize the datasets used in the experiments
# datasets are CIFAR10, CIFAR100 and Tiny ImageNet
# use create_val_folder() function to convert original Tiny ImageNet structure to structure PyTorch expects

import torch
import os
from torchvision import datasets, transforms, utils
from torch.utils.data import sampler, random_split, DataLoader
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data import Subset, ConcatDataset
from PIL import Image
import random
from helper_functions import CustomDataset 
import csv
import math  


def create_val_folder(root_dir):
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join(root_dir, 'tiny-imagenet-200/val/images')  # path where validation data is present now
    filename = os.path.join(root_dir, 'tiny-imagenet-200/val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create subfolders if not present, and move images into respective folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))


class TinyImagenet():
    def __init__(self, root_dir, batch_size=128, augment=True):
        self.name = "tinyimagenet"
        print('Loading TinyImageNet...')
        self.batch_size = batch_size
        self.img_size = 64
        self.num_classes = 200 #K  
        num_train = 100000
        self.num_test = 10000
        ratio_train = 0.9
        self.num_train = int(num_train * ratio_train)
        self.num_val = num_train - self.num_train
        
        train_dir = os.path.join(root_dir, 'tiny-imagenet-200/train')
        valid_dir = os.path.join(root_dir, 'tiny-imagenet-200/val/images')

        normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], 
                                         std =[0.2302, 0.2265, 0.2262])
        self.normalized = transforms.Compose([transforms.ToTensor(), 
                                              normalize])
        self.trainset = datasets.ImageFolder(train_dir, transform=self.normalized)
        
        train_ds, val_ds = random_split(self.trainset, [self.num_train, self.num_val],
                                        generator=torch.Generator().manual_seed(42))
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
        self.valid_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                                        num_workers=8)
        
        # Test
        self.testset = datasets.ImageFolder(valid_dir, transform=self.normalized)
        self.test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False,
                                                       num_workers=8)

        if augment: # if augment, change training dataloader 
            self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                                transforms.RandomCrop(64, padding=8),
                                                transforms.ColorJitter(0.2, 0.2, 0.2),
                                                transforms.ToTensor(), 
                                                normalize])    
            self.aug_trainset = datasets.ImageFolder(train_dir, transform=self.augmented)
            aug_train_ds, _ = random_split(self.aug_trainset,
                                           [self.num_train, self.num_val],
                                           generator=torch.Generator().manual_seed(42))
            self.train_loader = DataLoader(aug_train_ds, batch_size=batch_size, shuffle=True,
                                                                num_workers=8)
        
       
        # self.indices = list(range(self.num_test + self.num_val))
        # random.Random(0).shuffle(self.indices)
        # self.test_idx = self.indices[:self.num_test]
        # self.val_idx = self.indices[self.num_test:]
        # self.test_sampler = sampler.SubsetRandomSampler(self.test_idx)
        # self.val_sampler = sampler.SubsetRandomSampler(self.val_idx)
        # self.val_loader = DataLoader(self.testset, batch_size=batch_size,
        #                                               sampler=self.val_sampler, shuffle=False, num_workers=8)
        # self.test_loader = DataLoader(self.testset, batch_size=batch_size,
        #                                                sampler=self.test_sampler, shuffle=False, num_workers=8)


def have_dummy(row):
    s = row[0]+row[1]
    return ('dummy' in s)

def remove_dummy(tab):
    new_tab = []
    for row in tab:
        if have_dummy(row)==False:
            new_tab.append(row)
    return new_tab

def get_candidate_vague_classes(parent_to_subclasses, num_subs_low=2, num_subs_high=3):
    '''
    # superclasses_multiple
    # ['n04371563',
    #  'n03419014',
    #  'n04490091',
    #  'n02924116',
    #  'n02858304',
    #  'n04379243',
    # ]
    '''
    superclasses_multiple = [] # candidate superclasses 
    for superClass, subClasses in parent_to_subclasses.items():    
        if len(subClasses) in [num_subs_low, num_subs_high]:
            superclasses_multiple.append(superClass)
    return superclasses_multiple  # length: 29


def get_sample_idx_by_class(dataset, num_classes):
    '''
    Return: [[0,1,...,499],[500, 501,...,999], ..., [..., 99999]]
    200 lists of each 500 samples 
    '''
    sample_idx_by_class = [[] for i in range(num_classes)]
    for i in range(len(dataset)):
        sample_idx_by_class[dataset[i][1]].append(i)
    return sample_idx_by_class

def train_valid_ds(dataset, num_classes, perc=0):
    train_class_indices = []
    valid_class_indices = []
    sample_idx_by_class = get_sample_idx_by_class(dataset, num_classes)
    for samples in sample_idx_by_class:
        valid_samples = random.sample(samples, int(len(samples)*perc))
        valid_class_indices.append(valid_samples)
        train_class_indices.append(list(set(samples)-set(valid_samples)))
    valid_ds = Subset(dataset, valid_class_indices[0])
    train_ds = Subset(dataset, train_class_indices[0])
    for k in range(1, num_classes):
        valid_ds += Subset(dataset, valid_class_indices[k])
        train_ds += Subset(dataset, train_class_indices[k])
    return train_ds, valid_ds

def make_vague_samples(dataset, num_single, num_single_comp, vague_classes_ids, guass_blur_sigma = 15):
    sample_indices = [i for i in range(len(dataset))]
    num_samples_subclass = len([i for i in range(len(dataset)) if dataset[i][1] == 0]) #500 for TinyImageNet train   
    for k in range(num_single, num_single_comp): # i.e.: 200, 201
        num_vague = math.floor(num_samples_subclass / (len(vague_classes_ids[k-num_single]) + 1.0)) #todo: why this num?
        num_nonvague = num_samples_subclass - num_vague
        
        for subclass in vague_classes_ids[k - num_single]:
            idx_candidates = [i for i in range(len(dataset)) if dataset[i][1] == subclass]
            idx_non_candidates = list(set(sample_indices)-set(idx_candidates))
            subset1 = Subset(dataset, idx_non_candidates) 
            subset2 = Subset(dataset, idx_candidates)

            nonvague_samples, vague_samples = random_split(subset2, [num_nonvague, num_vague])
            # give new labels for vague images 
            vague_samples = CustomDataset(subset=vague_samples, 
                                        class_num=k, 
                                        transform=transforms.GaussianBlur(kernel_size=(35, 45), 
                                                                            sigma = guass_blur_sigma))
                                                            
            dataset = subset1 + nonvague_samples + vague_samples
            ### can we avoid changing the order of samples in dataset?
    return dataset

class tinyImageNetVague():
    def __init__(self, root_dir, batch_size=128, augment=True, num_comp=1, imagenet_hierarchy_path="./"):
        self.name = "tinyimagenet"
        print('Loading TinyImageNet...')
        self.batch_size = batch_size
        self.img_size = 64
        self.num_classes = 200 #K  
        self.num_comp = num_comp
        self.kappa = self.num_classes + self.num_comp
        num_train = 100000
        self.num_test = 10000
        ratio_train = 0.9
        self.num_train = int(num_train * ratio_train)
        self.num_val = num_train - self.num_train
        
        train_dir = os.path.join(root_dir, 'tiny-imagenet-200/train')
        val_img_dir = os.path.join(root_dir, 'tiny-imagenet-200/val/images')

        self.class_wnids = set() # class names in the format of nids: {'n01443537', 'n01629819', ... }
        for subclass in os.listdir(train_dir):
            self.class_wnids.add(subclass)
        
        preprocess_transform_pretrain = transforms.Compose([
                        transforms.Resize(256), # Resize images to 256 x 256
                        transforms.CenterCrop(224), # Center crop image
                        transforms.RandomHorizontalFlip()
        ])
        train_ds_original = datasets.ImageFolder(train_dir, transform=preprocess_transform_pretrain)
        test_ds = datasets.ImageFolder(val_img_dir, transform=preprocess_transform_pretrain)
        self.class_to_idx = train_ds_original.class_to_idx
        # classes_to_idx
        # {'n01443537': 0,
        #  'n01629819': 1,
        #  'n01641577': 2,
        #  ...
        # 'n12267677': 199}

        self.hierarchy = self.get_hierarchy(imagenet_hierarchy_path) 
        self.parent_to_subclasses = self.get_ParentToSubclasses()
        self.candidate_superclasses = get_candidate_vague_classes(self.parent_to_subclasses)
        self.vague_classes_nids, self.vague_classes_ids = get_vague_classes() 
        self.idx_to_class = {value:key for key, value in self.class_to_idx.items()}
        for i in range(self.num_comp):
            self.idx_to_class[self.num_classes + i] = [self.idx_to_class[k] for k in self.vague_classes_ids[i]]
        # {0: 'n01443537',
        # 1: 'n01629819',
        # 199: 'n12267677',
        # 200: ['n07871810', 'n07873807', 'n07875152']}

        train_ds = make_vague_samples(train_ds_original, self.num_classes, self.kappa, self.vague_classes_ids)
        test_ds = make_vague_samples(test_ds, self.num_classes, self.kappa, self.vague_classes_ids)
        train_ds, valid_ds = train_valid_ds(train_ds, self.num_classes, perc=1-ratio_train)

        print(f'Data splitted. training, validation, test size: {len(train_ds), len(valid_ds), len(test_ds)}')

        preprocess_normalization = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                            std =[0.229, 0.224, 0.225])])
        train_ds = CustomDataset(subset=train_ds, transform=preprocess_normalization)
        valid_ds = CustomDataset(subset=valid_ds, transform=preprocess_normalization)
        test_ds = CustomDataset(subset=test_ds, transform=preprocess_normalization)
        # # device = torch.device('cuda:0')
        # device = get_default_device()
        # print(device)
        self.train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=1, pin_memory=True)
        # train_dl = DeviceDataLoader(train_dl, device)

        self.valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=1, pin_memory=True)
        # valid_dl = DeviceDataLoader(valid_dl, device)

        self.test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=1, pin_memory=True)
        # test_dl = DeviceDataLoader(test_dl, device)

    def get_hierarchy(self, imagenet_hierarchy_path):
        #return hierarchy: size 1327*2 each row contains the information [superclass, subclass]
        '''
        [['n00001740', 'n00021939'],
        ['n00021939', 'n03051540'],
        ['n03419014', 'n04230808'],
        ['n04230808', 'n03534580'],
        ['n04230808', 'n03770439'],
        ['n04230808', 'n03866082'],
        ['n04230808', 'n04136333'],
        ...
        ]

        '''
        with open(imagenet_hierarchy_path + 'ImageNetHierarchy.txt') as f:
            reader = csv.reader(f, delimiter=' ')
            hierarchy = [line for line in reader]
        hierarchy = remove_dummy(hierarchy)
        return hierarchy 

    def get_ParentToSubclasses(self):
        '''
        # parent_to_subclass.items():
        # n04230808 :  ['n03770439']
        # n04370048 :  ['n02963159']
        # n04371563 :  ['n02837789', 'n04371430']
        # n03419014 :  ['n02730930', 'n03763968']
        # n03057021 :  ['n03404251', 'n02669723', 'n04532106', 'n03980874', 'n03617480']
        # n03314378 :  ['n03424325']
        '''
        parent_to_subclass = {}
        for superclass, subclass in self.hierarchy:
            if subclass in self.class_wnids:
                if superclass not in parent_to_subclass.keys():
                    parent_to_subclass[superclass] = [subclass]
                else:
                    parent_to_subclass[superclass].append(subclass)
        print("length of parent_to_subclass: ", len(parent_to_subclass))   # 107 

        # cnt_classes_rest = 0
        # for superClass, subclasses in parent_to_subclass.items():
        # #     print(superClass, ': ', subclasses)
        #     cnt_classes_rest += len(subclasses)
        # print("The num of subclasses in parent_to_subclass: ", cnt_classes_rest) # 175
        return parent_to_subclass
        
    def get_vague_classes(self):
        vague_classes = random.sample(self.candidate_superclasses, self.num_comp)
        print(f'randomly selecting {self.num_comp} vague superclasses from {len(self.candidate_superclasses)} \
                candidate superclasses:\n {vague_classes}')
        vague_subs_nids = [[self.parent_to_subclasses[super_class]] for super_class in vague_classes]
        vague_subs_ids = [[self.class_to_idx[sub] for sub in super] for super in vague_subs_nids]
        # C = [[classes_to_idx[sub_class] for sub_class in parent_to_subclass[super_class]] for super_class in vague_classes]
        return vague_subs_nids, vague_subs_ids

    

# The following are only for double checking  
def nIDs_to_names(DATA_DIR, class_wnids, parent_to_subclass):
    word_mapping = DATA_DIR + '/' + 'words.txt'
    with open(word_mapping) as f:
        tab_raw = f.readlines()
    wnID_to_names = {}  # contains the mapping from ID to names for all super/sub classes 
    for row in tab_raw:
        row = row.replace('\n','').split('\t')
        WNID = row[0]
        ImageLabels = row[1].split(', ')
        if WNID in class_wnids or WNID in parent_to_subclass:
            wnID_to_names[WNID] = ImageLabels

    print('number of WNIDs: ', len(wnID_to_names.keys())) 

    # one ID might -> multiple names  
    print("Example: ", wnID_to_names['n04370048'])
    # for class_wnID, class_name in wnID_to_names.items():
    #     print(class_wnID, ': ', class_name)

    # wnID_to_names.items():
    # n01443537 :  ['goldfish', 'Carassius auratus']
    # n01629276 :  ['salamander']
    # n01629819 :  ['European fire salamander', 'Salamandra salamandra']
    # n01639765 :  ['frog', 'toad', 'toad frog', 'anuran', 'batrachian', 'salientian']
    # n01640846 :  ['true frog', 'ranid']
    return wnID_to_names


def get_vague_classes(candidate_superclasses, num_comp, classes_to_idx, parent_to_subclass, num_singular):
    vague_classes = random.sample(candidate_superclasses, num_comp)
    print(f'randomly selecting {num_comp} vague classes from {len(candidate_superclasses)} candidate superclasses:\n {vague_classes}')

    C = [[classes_to_idx[sub_class] for sub_class in parent_to_subclass[super_class]] for super_class in vague_classes]
    kappa = num_singular + len(C)
    print(f'Composite set interget labels:C={C}, K = {num_singular}, kappa={kappa}')

    idx_to_class = {value:key for key, value in classes_to_idx.items()}

    for i in range(len(C)):
        idx_to_class[num_singular + i] = [idx_to_class[k] for k in C[i]]
    print('Interger class label to WNIDs:')

    print(f"subclasses in the vagued superclasses: {C}")
    for vague_class_id in range(num_singular, kappa):
        vague_class = vague_classes[vague_class_id - num_singular]
        print(f"Vague class idx:{vague_class_id}, wnID: {vague_class}: {parent_to_subclass[vague_class]}")
    
    # Vague class idx:200, wnID: n03880531: ['n03400231', 'n04596742']
    # Vague class idx:201, wnID: n01942177: ['n01944390', 'n01945685', 'n01950731']
    # Vague class idx:202, wnID: n04565375: ['n04008634', 'n02950826']
    # Vague class idx:203, wnID: n01772222: ['n01774384', 'n01774750']
    # Vague class idx:204, wnID: n03743902: ['n02892201', 'n04486054']

if __name__ == '__main__':

    # # double check 1
    # for superClass in parent_to_subclass:
    #     if superClass in tinyImageNet_class_wnids:
    #         print("Leaf node") # superClasses are not in the tinyImageNet_class_wnids

    # # double check 2
    # subclasses_list = []
    # for superclass, subClasses in parent_to_subclass.items():
    #     subclasses_list += subClasses
    # ### check if there is a subclasses with more than one superclass
    # print(len(subclasses_list))
    # subclasses_list  
    dataset = tinyImageNetVague()
    print(f"class_to_idx: {dataset.class_to_idx}")
    print(f"idx_to_class: {dataset.idx_to_class}")
    print(f"parent_to_subclasses: {dataset.parent_to_subclasses}")
    print(f"candicate superclasses: {dataset.candidate_superclasses}")

    
        # print(f"subclasses in the vagued superclasses: {C}")
        # for vague_class_id in range(num_singular, kappa):
        #     vague_class = vague_classes[vague_class_id - num_singular]
        #     print(f"Vague class idx:{vague_class_id}, wnID: {vague_class}: {parent_to_subclass[vague_class]}")
        
        # # Vague class idx:200, wnID: n03880531: ['n03400231', 'n04596742']
        # # Vague class idx:201, wnID: n01942177: ['n01944390', 'n01945685', 'n01950731']