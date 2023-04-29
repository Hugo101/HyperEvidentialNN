import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.dirichlet import Dirichlet
from collections import defaultdict


def logBetaFunc(values):
    '''
    beta function
    values: a list of values: []
    return log beta (.)
    '''
    # if len(values) == 1:
    #     return 1.
    # res = 0.
    # for value in values:
    #     res += torch.lgamma(value)
    # res = res - torch.lgamma(torch.sum(values))
    res = torch.sum(torch.lgamma(values)) - torch.lgamma(torch.sum(values))
    return res


class GroupDirichlet(ExponentialFamily):
    def __init__(self, concentration_a, concentration_b, partition_list, validate_args=False):
        '''
        concentration_a: torch.Tensor([a1, a2, a3, a4, a5]) #! not evidence
        concentration_b: torch.Tensor([b1, b2])
        partition_list:  [[1, 3, 0],[2, 4]]
        '''
        self.concentration_a = concentration_a
        self.concentration_b = concentration_b
        self.partition_list = partition_list
        self.n_partition = len(partition_list)
        
        assert len(self.concentration_b) == self.n_partition
        
        self.partition_cat_list = sum(self.partition_list, []) #[1, 3, 0, 2, 4]
        
        # two types of Dirichlet concentrations
        concentration_Y = []
        concentration_R = []
        for i in range(self.n_partition):
            parti_curr_ids = self.partition_list[i]
            concent_parti_cur = self.concentration_a[parti_curr_ids]
            concentration_Y.append(concent_parti_cur)
            concentration_R.append(torch.sum(concent_parti_cur) + self.concentration_b[i])
        
        self.concentration_Y = concentration_Y
        self.concentration_R = torch.tensor(concentration_R)
        
        super().__init__(validate_args=validate_args)


    def rsample(self, sample_shape=()):
        result = []
        dir_R_tmp = Dirichlet(self.concentration_R)
        R_samples = dir_R_tmp.sample(sample_shape)
        
        # get samples per partition 
        for i in range(self.n_partition):
            dir_Y_curr_part = Dirichlet(self.concentration_Y[i])
            Y_samples_curr = dir_Y_curr_part.sample(sample_shape)
            R_sample_curr = R_samples[:, i].unsqueeze(dim=1).expand(Y_samples_curr.shape)
            result.append(R_sample_curr*Y_samples_curr)
        
        result = torch.cat(result, dim=1)
        
        # recover the order
        #curr_sorted [1, 3, 0, 2, 4]
        #            [0, 1, 2, 3, 4]
        #    -> indx [2, 0, 3, 1, 4]
        _, indx = torch.sort(torch.tensor(self.partition_cat_list))
        
        return result.index_select(1, indx)


    def log_normalized_constant(self):
        '''
        self.concentration_Y: 
        self.concentration_R:
        return log(C)
        '''
        cg = 0.
        for i in range(self.n_partition):
            log_beta_value = logBetaFunc(self.concentration_Y[i])
            cg += log_beta_value
        cg += logBetaFunc(self.concentration_R)
        return cg


    def log_prob(self, value):
        '''
        value: probabilities of each singleton component, tensor:[p1, p2, p3, p4, p5, p6]
        '''
        term_1 = (torch.log(value) * (self.concentration_a - 1.0)).sum(-1)
        
        partition_prob_sums = []
        for i in range(self.n_partition):
            # partition_list [[1, 3, 0],[2, 4]]
            part_idx_curr = self.partition_list[i]
            prob_sum_part_curr = torch.sum(value[:, part_idx_curr], dim=-1, keepdim=True)
            partition_prob_sums.append(prob_sum_part_curr)
        partition_prob_sums = torch.cat(partition_prob_sums, dim=-1)
        term_2 = (torch.log(partition_prob_sums)*self.concentration_b).sum(-1)
        term_3 = self.log_normalized_constant()
        return term_1 + term_2 - term_3


    def entropy(self):
        log_Cg = self.log_normalized_constant()
        num_singles = self.concentration_a.size(-1)
        a0 = self.concentration_a.sum(-1)
        term2 = ((self.concentration_a - 1.0) * (torch.digamma(self.concentration_a))).sum(-1)
        
        beta = []
        partition_indicator = []
        for i in range(self.n_partition):
            # print(f"partition:{i}")
            part_idx_curr = self.partition_list[i]
            alpha_i = torch.sum(self.concentration_a[part_idx_curr])
            # print(alpha_sum, evidence_comps[i])
            beta_i = alpha_i + self.concentration_b[i]
            diff = torch.digamma(beta_i) - torch.digamma(alpha_i)
            partition_indicator.append(diff)
            beta.append(beta_i)
        beta_tensor = torch.stack(beta, dim=0) #! if using [] then tensor, the graident will lose
        beta0 = beta_tensor.sum(-1)
        term3 = (a0 - num_singles)*torch.digamma(beta0)
        
        term4 = 0.
        for k in range(num_singles):
            partition_id = find_partition(self.partition_list, k)
            term4 += partition_indicator[partition_id]*(self.concentration_a[k]-1)

        term5 = torch.sum(self.concentration_b * torch.digamma(beta_tensor))
        term6 = torch.digamma(beta0)*torch.sum(self.concentration_b)
        
        return log_Cg - term2 + term3 - term4 - term5 + term6


def find_partition(partition_list, class_id):
    n_pars = len(partition_list)
    for i in range(n_pars):
        partition = partition_list[i]
        if class_id in partition:
            return i


def numerator_GDD(probabilities, concentration_a, concentration_b, idx_comp_list):
    '''
    probabilities: [[p1, p2, p3, p4, p5],
                    [p1, p2, p3, p4, p5]]
    concentration_a: [10,  21, 32, 4, 5] #! not evidence
    concentration_b: [10, 7, 0]
    idx_comp_list: [[0, 3], [1, 4], [2]] 
    return: log(numerator)
    '''
    log_probs = torch.log(probabilities)
    part_1 = torch.sum(log_probs*(concentration_a-1), dim=1, keepdim=True)
    
    n_partition = len(idx_comp_list)
    
    part_2 = []
    for i in range(n_partition):
        part_idx_curr = idx_comp_list[i]
        tmp = torch.sum(probabilities[:, part_idx_curr], dim=1, keepdim=True)
        log_prob_tmp = torch.log(tmp)
        pow = concentration_b[i]
        part_2.append(log_prob_tmp*pow)
    part_2 = torch.cat(part_2, dim=1)
    part_2 = torch.sum(part_2, dim=1, keepdim=True)

    return part_1+part_2


def numerator_HDD(probabilities, concentration_a, concentration_b, idx_comp_list):
    '''
    probabilities: [[p1, p2, p3, p4, p5],
                    [p1, p2, p3, p4, p5]]
    concentration_a: [1,  2, 3, 4, 5] #! not evidence
    concentration_b: [10, 7, 4, 2]
    idx_comp_list: [[2,4], [1,5], [1,3,4], [2,3]] 
    '''
    log_probs = torch.log(probabilities)
    part_1 = torch.sum(log_probs*(concentration_a-1), dim=1, keepdim=True)
    
    n_partition = len(idx_comp_list)
    
    part_2 = []
    for i in range(n_partition):
        part_idx_curr = idx_comp_list[i]
        tmp = torch.sum(probabilities[:, part_idx_curr], dim=1, keepdim=True)
        log_prob_tmp = torch.log(tmp)
        pow = concentration_b[i]
        part_2.append(log_prob_tmp*pow)
    part_2 = torch.cat(part_2, dim=1)
    part_2 = torch.sum(part_2, dim=1, keepdim=True)

    return part_1+part_2


class HDD(object):
    def __init__(self, probabilities, concentration_a=None, concentration_b=None, idx_comp_list=None):
        self.probabilities = probabilities
        self.concentration_a = concentration_a
        self.concentration_b = concentration_b
        self.idx_comp_list = idx_comp_list


def weights_assigned(value, probabilities):
    '''
    value: a number, 5
    probabilities: a list of probabilities, [0.3, 0.2]
    return: [3, 2]
    '''
    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.tensor(probabilities)
    sum_probs = torch.sum(probabilities)
    return probabilities/sum_probs*value


class GDD_latentZ(object):
    def __init__(self, probabilities, concentr_a, concentr_b_partition,concentr_b_comp, partition_list, idx_comp_list) -> None:
        '''
        probabilities: [p1, p2, p3, p4, p5]
        concentr_a: [1,  2, 3, 4, 5]
        concentr_b_partition: [10, 0]
        concentr_b_comp: [7, 4, 2]
        partition: [[2,4], [1,3,5]]
        idx_comp_list: [[1,5], [1,3,4], [2,3]]
        (w.r.t. -> latent Z)
        '''
        self.probabilities = probabilities
        self.concentr_a = concentr_a
        self.n_prob = len(concentr_a)
        
        self.concentr_b_partition = concentr_b_partition
        self.partition_list = partition_list
        self.n_partition = len(partition_list)
        
        self.concentr_b_comp = concentr_b_comp
        self.idx_comp_list = idx_comp_list
        self.n_comp = len(idx_comp_list)


    def Estep(self):
        '''
        Perform an E(stimation)-step:
        '''
        latentZ = []
        for i in range(self.n_comp):
            parti_curr_ids = self.idx_comp_list[i]
            probs_curr = self.probabilities[parti_curr_ids]
            Z_curr = weights_assigned(self.concentr_b_comp[i], probs_curr)
            latentZ.append(Z_curr)
        self.latentZ = latentZ


    def Mstep(self):
        '''
        Perform an M(aximization)-step # compute denominators
        '''
        indx_z = defaultdict(lambda: 0)
        indx = sum(self.idx_comp_list, [])
        latentZ = torch.cat(self.latentZ, dim=0)
        for idx, z in zip(indx, latentZ):
            if idx not in indx_z:
                indx_z[idx] = z
            else:
                indx_z[idx] += z
        # idx: 1,2,3,4,5
        
        # calculate the mode
        # denominator: W
        # sumZ + sumA + sumB - n
        W = torch.sum(self.concentr_a) + torch.sum(self.concentr_b_comp) + torch.sum(self.concentr_b_partition) - len(self.probabilities) # todo: Check it!
        # W = torch.sum(self.concentr_a) + torch.sum(self.concentr_b_comp) + torch.sum(self.concentr_b_partition)
        # print(f"W: {W}")
        probabs = []
        for i in range(self.n_partition):
            part_curr_idx = self.partition_list[i]
            
            # term for current partition
            tmp_curr_part = 0
            for j in part_curr_idx:
                tmp_curr_part += indx_z[j]
                tmp_curr_part += self.concentr_a[j]
            tmp_curr_part -= len(part_curr_idx) ## todo: check this
            
            term_2 = 1 + self.concentr_b_partition[i]/tmp_curr_part
            # print(f"Partition {i}, term_2: {term_2}")
            
            # term for current prob in the partition
            for j in part_curr_idx:
                term_1 = indx_z[j] + self.concentr_a[j] - 1 ## todo: check this
                # term_1 = indx_z[j] + self.concentr_a[j]
                curr_p = term_1 /W * term_2
                probabs.append(curr_p)
        
        # recover the order
        self.partition_cat_list = sum(self.partition_list, [])
        _, indx = torch.sort(torch.tensor(self.partition_cat_list))
        probabs_tensor = torch.tensor(probabs)
        self.probabilities = probabs_tensor.index_select(0, indx)


    def iterate(self, n_iterations=5, verbose=True):
        '''
        Perform N iterations, then compute log-likelihood
        '''
        N = n_iterations
        for i in range(1, N+1):
            #The heart of the algorith, perform E-stepand next M-step
            self.Estep()
            # print(f'1 Latent Z: {self.latentZ}')
            self.Mstep()
            if verbose:
                print(f'\n Iteration: {i}')
                print(f'Latent Z: {self.latentZ}')
                print(f'Theta (probs): {self.probabilities}')
        
        self.Estep() # to freshen up self.loglike

        print("EM done")


    def add_Z_to_concentr_a(self):
        # acquire Z for each prob (idx)
        indx_z = defaultdict(lambda: 0)
        indx = sum(self.idx_comp_list, [])
        latentZ = torch.cat(self.latentZ, dim=0)
        for idx, z in zip(indx, latentZ):
            if idx not in indx_z:
                indx_z[idx] = z
            else:
                indx_z[idx] += z
        # idx: 1,2,3,4,5
        latentZ_list = []
        for i in range(self.n_prob):
            latentZ_list.append(indx_z[i])
        
        return self.concentr_a + torch.tensor(latentZ_list)


if __name__ == "__main__":
    
    ###! Basic Group Dirichlet Distribution sampling test
    # m = GroupDirichlet(torch.tensor([1., 2., 3., 4., 5.]), torch.tensor([5., 6.]), [[1,3,0],[2,4]])
    # s1 = m.sample((5,))
    # print(s1)
    
    ###! Group Dirichlet Distribution sampling test for 0 evidenece: m1 and m2 are the same
    m1 = GroupDirichlet(torch.tensor([1., 2., 3., 4., 5.]), torch.tensor([5., 0.]), [[1,3,0],[2,4]])
    log_Cg1 = m1.log_normalized_constant() 
    print(f"log_C_g: {log_Cg1}") #-22.7539
    s1 = m1.sample((5,))
    print(s1)
    ###
    #   ([[0.2286, 0.0543, 0.2092, 0.3239, 0.1840],
    #     [0.0311, 0.1919, 0.1363, 0.3737, 0.2669],
    #     [0.0772, 0.1910, 0.0263, 0.3177, 0.3878],
    #     [0.1209, 0.2139, 0.2589, 0.1283, 0.2781],
    #     [0.0437, 0.2883, 0.0858, 0.3254, 0.2568]])
    ###
    m2 = GroupDirichlet(torch.tensor([1., 2., 3., 4., 5.]), torch.tensor([5., 0., 0.]), [[1, 3, 0],[2],[4]])
    log_Cg2 = m2.log_normalized_constant()
    print(f"log_C_g: {log_Cg2}") #-22.7539
    s2 = m2.sample((5,))
    print(s2)
    
    ###! Dirichlet Distribution (Dir) for:
    ### 1) log_normalized_constant()
    ### 2) log_prob(value)
    evidence_single = torch.tensor([2., 6., 2.])
    m3 = GroupDirichlet(evidence_single+1, torch.tensor([0., 0.]), [[0, 1],[2]])
    log_Cg3 = m3.log_normalized_constant()
    print(f"log_Cg3: {log_Cg3}") # -12.0217 (checked correct)
    prob_tmp = torch.tensor([[0.2, 0.6, 0.2]])
    log_prob_m3 = m3.log_prob(prob_tmp)
    print(f"log_prob_m3: {log_prob_m3}") # tensor([2.5190]) (checked correct)
    
    Dir = Dirichlet(evidence_single+1)
    log_Cg_dir = Dir._log_normalizer(evidence_single+1)
    print(f"log_Cg_dir: {log_Cg_dir}") # -12.0217
    log_prob_dir = Dir.log_prob(prob_tmp)
    print(f"log_prob_Dir: {log_prob_dir}") # tensor([2.5190])
    
    log_numer_dir = log_prob_dir + log_Cg_dir
    print(f"log_numer_dir from Dirichlet Control: {log_numer_dir}") 
    
    log_numer_GDD = numerator_GDD(prob_tmp, evidence_single+1, torch.tensor([0., 0.]), [[0, 1],[2]])
    print(f"log_numer_GDD from GDD: {log_numer_GDD}") # tensor([-9.5027]) (checked correct)


###! Group Dirichlet Distribution (m3)
    ### 1) log_normalized_constant()
    ### 2) log_prob(value)
    evidence_single = torch.tensor([2., 6., 2.])
    m33 = GroupDirichlet(evidence_single+1, torch.tensor([2., 0.]), [[0, 1],[2]])
    log_Cg33 = m33.log_normalized_constant()
    print(f"log_Cg3: {log_Cg33}") # tensor(-12.5252) 
    prob_tmp = torch.tensor([[0.2, 0.6, 0.2]])
    log_prob_m33 = m33.log_prob(prob_tmp)
    print(f"log_prob_m33: {log_prob_m33}") # tensor([2.5762])
    
    log_numer_GDD_33 = numerator_GDD(prob_tmp, evidence_single+1, torch.tensor([2., 0.]), [[0, 1],[2]])
    print(f"log_numer_GDD from one way: {log_numer_GDD_33}") # tensor([-9.9490]) 

    log_numer_GDD_33_c1 = log_prob_m33 + log_Cg33
    print(f"log_numer_GDD from another way: {log_numer_GDD_33_c1}") # tensor([-9.9490]) (checked correct)
    
    ###! Dirichlet Distribution (m4) [checked correct]
    ###! Analytical differential Entropy vs. Sampled differential Entropy
    evidence_single = torch.tensor([2., 6., 2.])
    m4 = GroupDirichlet(evidence_single+1, torch.tensor([0., 0.]), [[0, 1],[2]])
    entropy_analytical_dir = m4.entropy()
    print(f"Analytical Differential Entropy: {entropy_analytical_dir}") # -1.6896 from analytical solution
    
    s4 = m4.sample((50000,))
    print(s4)
    entropy_est_dir = (-m4.log_prob(s4)).mean()
    print(f"Sampled Differential Entropy: {entropy_est_dir}") # -1.6856 from sampled Shannon Entropy solution
    
    ###! sampled Shannon Entropy
    p_log_p = -s4*torch.log(s4)
    entropy_est_4 = p_log_p.sum(dim=1).mean()
    print(entropy_est_4) # 0.9379 from sampled Shannon Entropy solution


    ###! Group Dirichlet Distribution (m5) [checked correct]
    ###! Analytical differential Entropy vs. Sampled differential Entropy
    evidence_single = torch.tensor([2., 6., 2.])
    m5 = GroupDirichlet(evidence_single+1, torch.tensor([2., 0.]), [[0, 1],[2]])
    # m5 = GroupDirichlet(evidence_single+1, torch.tensor([2., 2.]), [[0, 1],[2]])
    entropy_analytical_gdd = m5.entropy()
    print(f"Analytical Differential Entropy: {entropy_analytical_gdd}") # -1.7735 from analytical solution
    # print(f"Analytical Differential Entropy: {entropy_analytical_gdd}") # -1.7911 from analytical solution
    
    #! Sampled Differential Entropy
    s5 = m5.sample((50000,))
    entropy_est_gdd = (-m5.log_prob(s5)).mean()
    print(f"Sampled Differential Entropy: {entropy_est_gdd}") # -1.7708 from sampled solution

    #! sampled Shannon Entropy
    p_log_p = -s5*torch.log(s5)
    entropy_est_5 = p_log_p.sum(dim=1).mean()
    print(entropy_est_5) # 0.9202 from sampled Shannon Entropy solution
    
    evidence_single = torch.tensor([2., 6., 2., 9., 1., 5.])
    m5 = GroupDirichlet(evidence_single+1, torch.tensor([12., 34., 0.]), [[0, 1],[2, 4], [3, 5]])
    entropy_analytical_gdd = m5.entropy()
    print(f"Analytical Differential Entropy: {entropy_analytical_gdd}") # -8.4025 from analytical solution
    
    #! Sampled Differential Entropy
    s5 = m5.sample((50000,))
    entropy_est_gdd = (-m5.log_prob(s5)).mean()
    print(f"Sampled Differential Entropy: {entropy_est_gdd}") # -8.4049 from sampled solution

    ###! EM Test
    probabilities = torch.tensor([1/4]*4)
    evidence_single = torch.tensor([6,3,8,8])
    concentr_a = evidence_single + 1
    concentr_b_partition = torch.tensor([2,4])
    concentr_b_comp = torch.tensor([2,0])
    partition_list = [[0,1], [2,3]]
    idx_comp_list = [[0,2], [1,3]]
    GDD_Z = GDD_latentZ(probabilities, concentr_a, concentr_b_partition, concentr_b_comp, partition_list, idx_comp_list)
    n_iterations = 4
    GDD_Z.iterate(n_iterations=n_iterations)
    
    
    ###! importance sampling (normalizing constant)
    probabilities = torch.tensor([1/3]*3)
    evidence_single = torch.tensor([30, 36, 22])
    concentr_a = evidence_single + 1
    concentr_b_partition = torch.tensor([35, 0])
    concentr_b_comp = torch.tensor([35,18])
    partition_list = [[0,1], [2]]
    idx_comp_list = [[1,2], [0,2]]
    GDD_Z = GDD_latentZ(probabilities, concentr_a, concentr_b_partition, concentr_b_comp, partition_list, idx_comp_list)
    n_iterations = 10
    GDD_Z.iterate(n_iterations=n_iterations)
    
    # Iteration: 10
    # Latent Z: [tensor([22.9485, 12.0515]), tensor([10.1652,  7.8348])]
    # Theta (probs): tensor([0.3088, 0.4532, 0.2380])
    
    addZ2a = GDD_Z.add_Z_to_concentr_a()
    
    ###! GDD normalizing constant
    concentr_a = addZ2a
    concentr_b = torch.tensor([35, 0])
    partition_list =  [[0,1], [2]]
    GDD = GroupDirichlet(concentr_a, concentr_b, partition_list, validate_args=None)
    log_Cg = GDD.log_normalized_constant()
    print(log_Cg) # tensor(4.4914e-30) * tensor(5.0587e-43)
    
    n_sample = 100
    probab_sampled = GDD.sample((n_sample, ))
    numer_GDD = numerator_GDD(probab_sampled, concentr_a, concentr_b, partition_list)
    
    concentr_a_old = torch.tensor([30, 36, 22])
    concentr_b_old = torch.tensor([35, 0, 35, 18])
    idx_comp_list = [[0,1], [2], [1,2], [0,2]]
    numer_HDD = numerator_HDD(probab_sampled, concentr_a_old, concentr_b_old, idx_comp_list)

    log_C_hdd_estimated = torch.log((numer_HDD - numer_GDD).exp().mean()) + log_Cg
    print(log_C_hdd_estimated)
    
    # HDD test case
    
    
    # GDD test case