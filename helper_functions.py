import torch 
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from sklearn import metrics

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

class CustomDataset(Dataset):
    def __init__(self, subset, class_num=None, transform=None):
        self.subset = subset
        self.transform = transform
        self.class_num = class_num
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        if self.class_num is not None:
            return x, self.class_num
        else:
            return x, y
        
    def __len__(self):
        return len(self.subset)

# not useful?
def get_partitions(num_single, vague_classes_ids):
    remaining_indices = list(range(0, num_single))
    partitions = []
    for comp_set in vague_classes_ids:
        partitions.append(comp_set)
        for i in comp_set:
            remaining_indices.remove(i) 
    if len(remaining_indices):
        partitions.append(remaining_indices)
    return partitions

'''
partition for GDD:
[[87, 92, 141], [34, 58, 33], [22, 11], [190, 191, 192], [91, 76], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 193, 194, 195, 196, 197, 198, 199]]
'''


def meanGDD(partitions, alpha, r, num_single, num_comp, device):
    # Probably from page 102 in the book 
    num_partitions = len(partitions)

    beta = torch.zeros(len(alpha), num_partitions).to(device)

    alpha_sum = torch.zeros(len(alpha), num_comp).to(device)

    for l in range(num_comp):
        alpha_sum[:, l] = torch.sum(torch.index_select(alpha, 1, torch.tensor(partitions[l]).to(device)), dim = 1)
        beta[:, l] = alpha_sum[:, l] + r[:, num_single + l]

    if num_partitions > num_comp:
        beta[:, num_comp] = torch.sum(torch.index_select(alpha, 1, torch.tensor(partitions[num_comp]).to(device)), dim = 1)

    p = torch.zeros(len(alpha), num_single).to(device)

    beta_sum = torch.sum(beta, dim=1)

    for l in range(num_comp):
        for k in partitions[l]:
            p[:, k] = (alpha[:, k] / beta_sum) * (beta[:, l] / alpha_sum[:, l])

    if num_partitions > num_comp:
        for k in partitions[num_comp]:
            p[:, k] = alpha[:, k] / beta_sum

    return p


def numAccurate(r, labels, num_single, W, R, a):
    alpha = torch.add(r[:,:num_single], torch.mul(W, a))

    # Get the predicted labels
    p_exp = meanGDD(alpha, r)
    predicted_labels = torch.argmax(p_exp, dim=1)

    total_correct = 0.0
    for i in range(len(labels)):
        predicted_set = set(R[torch.argmax(r[i])])

        if len(predicted_set) == 1:
            predicted_set = set(R[predicted_labels[i].item()])

        ground_truth_set = set(R[labels[i]])
        total_correct += float(len(predicted_set.intersection(ground_truth_set))) / len(predicted_set.union(ground_truth_set))

    return total_correct

# todo 
def vague_belief_mass(b, K, C, R, a_copy, device):
    b_v = torch.zeros(len(b), K).to(device)
    sum_beliefs = torch.zeros(len(b), K).to(device)

    for k in range(K):
        for l in range(len(C)):
            relative_base_rate = torch.zeros(1).to(device)
            intersection_set = set(R[k]).intersection(set(C[l]))
            if len(intersection_set) > 0:
                relative_base_rate = a_copy[R.index(list(intersection_set))] / a_copy[K + l]
            sum_beliefs[:, k] = sum_beliefs[:, k] + relative_base_rate * b[:, K + l]
        b_v[:, k] = sum_beliefs[:,k]
    
    return b_v


# def calculate_metrics(output, labels, K, W, a, R):
#     correct_vague = 0.0
#     correct_nonvague = 0.0
#     vague_total = 0
#     nonvague_total = 0
    
#     alpha = torch.add(output[:,:K], torch.mul(W, a)) # [64,200], unnormalized

#     # Get the predicted labels
#     p_exp = meanGDD(alpha, output) # [64,200], normalized, ??? why 200? 
#     predicted_labels = torch.argmax(p_exp, dim=1)

#     # Calculate vaguenesses (not used?)
#     b = output / (torch.sum(output, dim=1) + W)[:, None]
#     total_vaguenesses = torch.sum(b[:, K:], dim=1)
#     b_v = vague_belief_mass(b)

#     for i in range(len(labels)): # a batch of examples
#         predicted_set = set(R[torch.argmax(r[i])])
#         if len(predicted_set) == 1: #???? 
#             predicted_set = set(R[predicted_labels[i].item()])
        
#         ground_truth_set = set(R[labels[i].item()])
        
#         inter_set = predicted_set.intersection(ground_truth_set)
#         union_set = predicted_set.union(ground_truth_set)
#         rate = float(len(inter_set)/len(union_set))
        
#         if len(predicted_set) == 1:
#             correct_nonvague += rate
#             nonvague_total += 1
#         else:
#             correct_vague += rate 
#             vague_total += 1
      
#     return [correct_nonvague, correct_vague, nonvague_total, vague_total]