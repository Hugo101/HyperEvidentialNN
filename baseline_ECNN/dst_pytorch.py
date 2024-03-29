import torch
import numpy as np
from efficientnet_pytorch import EfficientNet
from torch import nn, optim


class Distance_layer(torch.nn.Module):
    '''
    verified
    '''
    def __init__(self, n_prototypes, n_feature_maps):
        super(Distance_layer, self).__init__()
        self.w = torch.nn.Linear(in_features=n_feature_maps, out_features=n_prototypes, bias=False).weight
        self.n_prototypes = n_prototypes

    def forward(self, inputs):
        for i in range(self.n_prototypes):
            if i == 0:
                un_mass_i = (self.w[i, :] - inputs) ** 2
                un_mass_i = torch.sum(un_mass_i, dim=-1, keepdim=True)
                un_mass = un_mass_i

            if i >= 1:
                un_mass_i = (self.w[i, :] - inputs) ** 2
                un_mass_i = torch.sum(un_mass_i, dim=-1, keepdim=True)
                un_mass = torch.cat([un_mass, un_mass_i], -1)
        return un_mass


class DistanceActivation_layer(torch.nn.Module):
    '''
    verified
    '''
    def __init__(self, n_prototypes,init_alpha=0,init_gamma=0.1):
        super(DistanceActivation_layer, self).__init__()
        self.eta = torch.nn.Linear(in_features=n_prototypes, out_features=1, bias=False)#.weight.data.fill_(torch.from_numpy(np.array(init_gamma)).to(device))
        self.xi = torch.nn.Linear(in_features=n_prototypes, out_features=1, bias=False)#.weight.data.fill_(torch.from_numpy(np.array(init_alpha)).to(device))
        torch.nn.init.kaiming_uniform_(self.eta.weight)
        torch.nn.init.kaiming_uniform_(self.xi.weight)
        # torch.nn.init.constant_(self.eta.weight,init_gamma)
        # torch.nn.init.constant_(self.xi.weight,init_alpha)
        #self.alpha_test = 1/(torch.exp(-self.xi.weight)+1)
        self.n_prototypes = n_prototypes
        self.alpha = None

    def forward(self, inputs):
        gamma=torch.square(self.eta.weight)
        alpha=torch.neg(self.xi.weight)
        alpha=torch.exp(alpha)+1
        alpha=torch.div(1, alpha)
        self.alpha=alpha
        si=torch.mul(gamma, inputs)
        si=torch.neg(si)
        si=torch.exp(si)
        si = torch.mul(si, alpha)
        max_val, max_idx = torch.max(si, dim=-1, keepdim=True)
        si /= (max_val + 0.0001)

        return si


'''class Belief_layer(torch.nn.Module):
    def __init__(self, prototypes, num_class):
        super(DS2, self).__init__()
        self.beta = torch.nn.Linear(in_features=prototypes, out_features=num_class, bias=False).weight
        self.prototypes = prototypes
        self.num_class = num_class

    def forward(self, inputs):
        beta = torch.square(self.beta)
        beta_sum = torch.sum(beta, dim=0, keepdim=True)
        self.u = torch.div(beta, beta_sum)
        inputs_new = torch.unsqueeze(inputs, dim=-2)
        for i in range(self.prototypes):
            if i == 0:
                mass_prototype_i = torch.mul(self.u[:, i], inputs_new[..., i])  #batch_size * n_class
                mass_prototype = torch.unsqueeze(mass_prototype_i, -2)
            if i > 0:
                mass_prototype_i = torch.unsqueeze(torch.mul(self.u[:, i], inputs_new[..., i]), -2)
                mass_prototype = torch.cat([mass_prototype, mass_prototype_i], -2)
        return mass_prototype'''

class Belief_layer(torch.nn.Module):
    '''
    verified
    '''
    def __init__(self, n_prototypes, num_class):
        super(Belief_layer, self).__init__()
        self.beta = torch.nn.Linear(in_features=n_prototypes, out_features=num_class, bias=False).weight
        self.num_class = num_class
    def forward(self, inputs):
        beta = torch.square(self.beta)
        beta_sum = torch.sum(beta, dim=0, keepdim=True)
        u = torch.div(beta, beta_sum)
        mass_prototype = torch.einsum('cp,b...p->b...pc',u, inputs)
        return mass_prototype

class Omega_layer(torch.nn.Module):
    '''
    verified, give same results

    '''
    def __init__(self, n_prototypes, num_class):
        super(Omega_layer, self).__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        mass_omega_sum = 1 - torch.sum(inputs, -1, keepdim=True)
        #mass_omega_sum = 1. - mass_omega_sum[..., 0]
        #mass_omega_sum = torch.unsqueeze(mass_omega_sum, -1)
        mass_with_omega = torch.cat([inputs, mass_omega_sum], -1)
        return mass_with_omega

class Dempster_layer(torch.nn.Module):
    '''
    verified give same results

    '''
    def __init__(self, n_prototypes, num_class):
        super(Dempster_layer, self).__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        m1 = inputs[..., 0, :]
        omega1 = torch.unsqueeze(inputs[..., 0, -1], -1)
        for i in range(self.n_prototypes - 1):
            m2 = inputs[..., (i + 1), :]
            omega2 = torch.unsqueeze(inputs[..., (i + 1), -1], -1)
            combine1 = torch.mul(m1, m2)
            combine2 = torch.mul(m1, omega2)
            combine3 = torch.mul(omega1, m2)
            combine1_2 = combine1 + combine2
            combine2_3 = combine1_2 + combine3
            combine2_3 = combine2_3 / torch.sum(combine2_3, dim=-1, keepdim=True)
            m1 = combine2_3
            omega1 = torch.unsqueeze(combine2_3[..., -1], -1)
        return m1


class DempsterNormalize_layer(torch.nn.Module):
    '''
    verified

    '''
    def __init__(self):
        super(DempsterNormalize_layer, self).__init__()
    def forward(self, inputs):
        mass_combine_normalize = inputs / torch.sum(inputs, dim=-1, keepdim=True)
        return mass_combine_normalize


class Dempster_Shafer_module(torch.nn.Module):
    def __init__(self, n_feature_maps, n_classes, n_prototypes):
        super(Dempster_Shafer_module, self).__init__()
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.n_feature_maps = n_feature_maps
        self.ds1 = Distance_layer(n_prototypes=self.n_prototypes, n_feature_maps=self.n_feature_maps)
        self.ds1_activate = DistanceActivation_layer(n_prototypes = self.n_prototypes)
        self.ds2 = Belief_layer(n_prototypes= self.n_prototypes, num_class=self.n_classes)
        self.ds2_omega = Omega_layer(n_prototypes= self.n_prototypes,num_class= self.n_classes)
        self.ds3_dempster = Dempster_layer(n_prototypes= self.n_prototypes,num_class= self.n_classes)
        self.ds3_normalize = DempsterNormalize_layer()

    def forward(self, inputs):
        '''
        '''
        ED = self.ds1(inputs)
        ED_ac = self.ds1_activate(ED)
        mass_prototypes = self.ds2(ED_ac)
        mass_prototypes_omega = self.ds2_omega(mass_prototypes)
        mass_Dempster = self.ds3_dempster(mass_prototypes_omega)
        mass_Dempster_normalize = self.ds3_normalize(mass_Dempster)
        return mass_Dempster_normalize



def tile(a, dim, n_tile, device):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        device)
    return torch.index_select(a, dim, order_index)


class DM(torch.nn.Module):
    def __init__(self, num_class, nu=0.9, device=torch.device('cpu')):
        super(DM, self).__init__()
        self.nu = nu
        self.num_class = num_class
        self.device = device

    def forward(self, inputs):
        upper = torch.unsqueeze((1 - self.nu) * inputs[..., -1], -1)  # here 0.1 = 1 - \nu
        upper = tile(upper, dim=-1, n_tile=self.num_class + 1, device=self.device)
        outputs = (inputs + upper)[..., :-1]
        return outputs


class DM_set_test(nn.Module):
    '''
    '''
    def __init__(self, num_class, num_set, nu):
        super(DM_set_test, self).__init__()
        self.num_class = num_class 
        self.nu = nu 
        self.utility_matrix = nn.Linear(in_features = num_class, out_features = num_set, bias=False).weight 
    
    def forward(self, inputs):
        for i in range(len(self.utility_matrix)): 
            if i == 0:
                precise = torch.mul(inputs[:, 0: self.num_class], self.utility_matrix[i])
                precise = torch.sum(precise, dim=-1, keepdim=True) #dim=0, 1 ?
                
                omega_1 = torch.mul(inputs[:, -1], torch.max(self.utility_matrix[i]))
                omega_2 = torch.mul(inputs[:, -1], torch.min(self.utility_matrix[i]))
                
                omega = torch.unsqueeze(self.nu*omega_1+(1-self.nu)*omega_2, -1)
                omega = omega.type(torch.float32)
                utility = precise + omega
                
            if i >= 1:
                precise = torch.mul(inputs[:, 0: self.num_class], self.utility_matrix[i])
                precise = torch.sum(precise, dim=-1, keepdim=True) #dim=0, 1 ?
                
                omega_1 = torch.mul(inputs[:, -1], torch.max(self.utility_matrix[i]))
                omega_2 = torch.mul(inputs[:, -1], torch.min(self.utility_matrix[i]))
                
                omega = torch.unsqueeze(self.nu*omega_1+(1-self.nu)*omega_2, -1)
                omega = omega.type(torch.float32)
                utility_i = precise + omega
                utility = torch.cat([utility, utility_i], -1)
                
        return utility 


## backbones for ECNN
class EfficientNet_DS(nn.Module):
    def __init__(self, n_feature_maps, n_classes, n_prototypes):
        super().__init__()
        
        self.n_feature_maps= n_feature_maps 
        self.n_classes = n_classes 
        self.n_prototypes = n_prototypes

        # EfficientNet
        self.network = EfficientNet.from_pretrained("efficientnet-b3", num_classes = self.n_classes)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self.network._global_params.dropout_rate)
        
        self.dempster_shafer = Dempster_Shafer_module(self.n_feature_maps, self.n_classes, self.n_prototypes)
        
    def forward(self, x):
        x = self.network.extract_features(x)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        # x = self._dropout(x)
        ## DS layers 
        x = self.dempster_shafer(x)
        #Utility layer for training
#         outputs = self.DM(x)
        return x




class SVHNCnnModel_DS(nn.Module):
    def __init__(self, n_feature_maps, n_classes, n_prototypes, dropout=False):
        super(SVHNCnnModel_DS, self).__init__()
        self.use_dropout = dropout
        
        self.n_feature_maps= n_feature_maps 
        self.n_classes = n_classes 
        self.n_prototypes = n_prototypes
        
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 256 x 4 x 4 (output: 256 x 3 x 3)

            nn.Flatten(), #  output: 256 x 3 x 3
            # nn.Linear(256*3*3, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(512, 10)
            )
    
        self.dempster_shafer = Dempster_Shafer_module(self.n_feature_maps, self.n_classes, self.n_prototypes)
    
    def forward(self, xb):
        x = self.network(xb)
        x = self.dempster_shafer(x)
        return x