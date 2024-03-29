# to standardize the datasets used in the experiments
# datasets are TinyImageNet, and CIFAR100 (later)
# use create_val_folder() function to convert original Tiny ImageNet structure to structure PyTorch expects

import os
import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import sampler, random_split, DataLoader
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import Subset, ConcatDataset
from PIL import Image
import random
from helper_functions import CustomDataset, AddLabelDataset
import csv
import math  
import time
from sklearn.model_selection import train_test_split
from collections import defaultdict


def create_val_folder(data_dir):
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join(data_dir, 'tiny-imagenet-200/val/images')  # path where validation data is present now
    filename = os.path.join(data_dir, 'tiny-imagenet-200/val/val_annotations.txt')  # file where image2class mapping is present
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


class tinyImageNetOrig():
    def __init__(self, data_dir, batch_size=128, num_workers=4, add_label=False):
        self.name = "tinyimagenet"
        print('Loading the original TinyImageNet...')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = 64
        self.num_classes = 200 #K  
        num_train = 100000
        self.num_test = 10000
        ratio_train = 0.9
        self.num_train = int(num_train * ratio_train)
        self.num_val = num_train - self.num_train
        self.vague_classes_ids = [[73, 135], [145, 166], [74, 158], [15, 16, 17], [185, 183], [179, 180], [48, 50], [108, 115], [30, 31, 32], [59, 75]]
        self.R = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83], [84], [85], [86], [87], [88], [89], [90], [91], [92], [93], [94], [95], [96], [97], [98], [99], [100], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111], [112], [113], [114], [115], [116], [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131], [132], [133], [134], [135], [136], [137], [138], [139], [140], [141], [142], [143], [144], [145], [146], [147], [148], [149], [150], [151], [152], [153], [154], [155], [156], [157], [158], [159], [160], [161], [162], [163], [164], [165], [166], [167], [168], [169], [170], [171], [172], [173], [174], [175], [176], [177], [178], [179], [180], [181], [182], [183], [184], [185], [186], [187], [188], [189], [190], [191], [192], [193], [194], [195], [196], [197], [198], [199], [73, 135], [145, 166], [74, 158], [15, 16, 17], [185, 183], [179, 180], [48, 50], [108, 115], [30, 31, 32], [59, 75]]
        self.num_classes = 200 #K  
        self.num_comp = 10
        self.kappa = self.num_classes + self.num_comp
        
        train_dir = os.path.join(data_dir, 'tiny-imagenet-200/train')
        valid_dir = os.path.join(data_dir, 'tiny-imagenet-200/val/images')

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
        
        self.trainset = datasets.ImageFolder(train_dir, transform=pre_norm_train)
        
        train_ds, valid_ds = random_split(self.trainset, [self.num_train, self.num_val],
                                        generator=torch.Generator().manual_seed(42))
        
        # Test
        testset = datasets.ImageFolder(valid_dir, transform=pre_norm_test)

        if add_label:
            train_ds = AddLabelDataset(train_ds) #add an aditional label
            valid_ds = AddLabelDataset(valid_ds)
            testset  = AddLabelDataset(testset)

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
        self.valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=8)
        self.test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

        

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
    gauss_kernel_size=5, data_train=True):
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

    if data_train:
        num_samples_subclass = 900 # 450 for TinyImageNet train
    else:
        num_samples_subclass = 100  # 50 for TinyImageNet valid / test
    
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



class tinyImageNetVague():
    def __init__(
        self, 
        data_dir, 
        num_comp=1, 
        batch_size=128, 
        ratio_train=0.9, 
        imagenet_hierarchy_path="./",
        duplicate=False,
        blur=True,
        gray=False,
        gauss_kernel_size=5,
        pretrain=True, #if using pretrained model, resize img to 224
        num_workers=4, #tune this on different servers
        seed=42):
        self.name = "tinyimagenet"
        print('Loading TinyImageNet...')
        start_time = time.time()
        self.blur = blur
        self.gray = gray
        self.gauss_kernel_size = gauss_kernel_size
        self.duplicate = duplicate
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.num_workers = num_workers
        self.img_size = 64
        self.num_classes = 200 #K  
        self.num_comp = num_comp
        self.kappa = self.num_classes + self.num_comp
        num_train = 100000
        self.num_train = int(num_train * ratio_train)
        self.num_val = num_train - self.num_train
        self.num_test = 10000

        train_dir = os.path.join(data_dir, 'tiny-imagenet-200/train')
        val_img_dir = os.path.join(data_dir, 'tiny-imagenet-200/val/images')
        train_ds_original = datasets.ImageFolder(train_dir)
        test_ds_original = datasets.ImageFolder(val_img_dir)
        self.class_to_idx = train_ds_original.class_to_idx
        # classes_to_idx
        # {'n01443537': 0,
        #  'n01629819': 1,
        #  ...
        # 'n12267677': 199}
        
        # class names in the format of nids: {'n01443537', 'n01629819', ... }
        self.class_wnids = set()
        for subclass in os.listdir(train_dir):
            self.class_wnids.add(subclass)

        self.hierarchy = self.get_hierarchy(imagenet_hierarchy_path) 
        self.parent_to_subclasses = self.get_ParentToSubclasses()
        self.candidate_superclasses = get_candidate_vague_classes(self.parent_to_subclasses)
        print(f"Total {len(self.candidate_superclasses)} Candidate superclasses: {self.candidate_superclasses}")
        self.vague_classes_nids, self.vague_classes_ids = self.get_vague_classes_v2() 
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
            gray=self.gray,
            gauss_kernel_size=self.gauss_kernel_size,
            data_train=True)
        valid_ds = make_vague_samples(
            valid_ds_original_n,
            self.num_classes, self.kappa,
            self.vague_classes_ids,
            blur=self.blur,
            gray=self.gray,
            gauss_kernel_size=self.gauss_kernel_size,
            data_train=False)
        test_ds = make_vague_samples(
            test_ds_original_n,
            self.num_classes, self.kappa,
            self.vague_classes_ids,
            blur=self.blur,
            gray=self.gray,
            gauss_kernel_size=self.gauss_kernel_size,
            data_train=False)
        
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
            norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
        with open(os.path.join(imagenet_hierarchy_path,'ImageNetHierarchy.txt')) as f:
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


    def get_vague_classes_v2(self):
        # vague_classes = random.sample(self.candidate_superclasses, self.num_comp)
        if self.num_comp == 1:
            vague_classes = ["n03419014"] #todo: fixed this for now
        else:
            vague_classes = random.sample(self.candidate_superclasses, self.num_comp)
        vague_subs_nids = [self.parent_to_subclasses[super_class] for super_class in vague_classes]
        vague_subs_ids = [[self.class_to_idx[sub] for sub in super] for super in vague_subs_nids]
        # C = [[classes_to_idx[sub_class] for sub_class in parent_to_subclass[super_class]] for super_class in vague_classes]
        
        print(f'Vague classes: {vague_classes}')
        return vague_subs_nids, vague_subs_ids
    
    
    def modify_vague_samples_old(self, dataset):
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
    ##### For further double checking, please check the corresponding notebook
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