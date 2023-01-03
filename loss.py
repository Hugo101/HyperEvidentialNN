import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet
from common_tools import set_device

kl_loss = nn.KLDivLoss(reduction="batchmean")

### ### The followings are for ENN mainly
def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = set_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = set_device()
    # y = y.to(device)
    # alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(
    y, alpha, num_classes, 
    kl_reg=True, 
    device=None):
    if not device:
        device = set_device()
    # y = y.to(device)
    # alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)
    ll_mean = torch.mean(loglikelihood)
    
    if not kl_reg:
        return ll_mean, 0

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_term = kl_divergence(kl_alpha, num_classes, device=device)
    kl_mean = torch.mean(kl_term)
    return ll_mean, kl_mean


def edl_loss(
    func, y, alpha, num_classes, 
    kl_reg=True, 
    device=None):
    # y = y.to(device)
    # alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    A_mean = torch.mean(A)

    if not kl_reg:
        return A_mean, 0
    
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_term = kl_divergence(kl_alpha, num_classes, device=device)
    kl_mean = torch.mean(kl_term)
    return A_mean, kl_mean


def edl_mse_loss(
    output, target, epoch_num, num_classes, 
    annealing_step, kl_lam, 
    kl_lam_teacher, 
    entropy_lam, 
    anneal=False, 
    kl_reg=True,
    kl_reg_teacher=False,
    entropy_reg=False, 
    device=None):
    if not device:
        device = set_device()
    if not kl_reg:
        assert anneal == False
    # evidence = relu_evidence(output)
    evidence = output
    alpha = evidence + 1
    ll_mean, kl_mean = mse_loss(
        target, alpha, num_classes, 
        kl_reg=kl_reg, 
        device=device
        )

    if not kl_reg:
        return ll_mean, ll_mean.detach(), 0
    
    if anneal:
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )    
        kl_div = annealing_coef * kl_mean
    else:
        kl_div = kl_lam * kl_mean

    loss = ll_mean + kl_div
    return loss, ll_mean.detach(), kl_mean.detach()


def edl_log_loss(
    output, target, epoch_num, num_classes, 
    annealing_step, kl_lam,
    anneal=False, 
    kl_reg=True, 
    device=None):
    if not device:
        device = set_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, 
            annealing_step, kl_lam, 
            anneal=anneal, kl_reg=kl_reg, 
            device=device
        )
    )
    return loss


# Expected Cross Entropy 
def edl_digamma_loss(
    output, target, epoch_num, num_classes,
    annealing_step, kl_lam,
    kl_lam_teacher,
    entropy_lam,
    ce_lam,
    pretrainedProb,
    forward,
    anneal=False,
    kl_reg=True,
    kl_reg_teacher=False,
    entropy_reg=False,
    exp_type=2,
    device=None
):
    if not device:
        device = set_device()
    if not kl_reg:
        assert anneal == False
    # evidence = relu_evidence(output)
    evidence = output
    alpha = evidence + 1

    
    ll_mean, kl_mean = edl_loss(
        torch.digamma, target, alpha, num_classes, 
        kl_reg=kl_reg, 
        device=device
        )

    if anneal:
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32, device=device),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32, device=device),
        )
        kl_div = annealing_coef * kl_mean
    else:
        kl_div = kl_lam * kl_mean
        
    # loss = ll_mean + kl_div
    
    # Cross Entropy calculated based on expected class probabilities
    if exp_type == 2:
        ce = nll(alpha, target)
        loss = ll_mean + kl_div + ce_lam*ce
        return loss, ll_mean.detach(), kl_mean.detach(), ce.detach() 
    
    #  KL divergence between 
    # expected class probabilities and 
    # the class probabilities predicted by detNN 
    if exp_type == 3:
        kl = KL_expectedProbSL_teacherProb(alpha, pretrainedProb, forward=forward)
        loss = ll_mean + kl_div + kl_lam_teacher * kl
        return loss, ll_mean.detach(), kl_mean.detach(), kl.detach()
    
    # Entropy
    if exp_type == 4:
        entropy = entropy_SL(alpha)
        loss = ll_mean - entropy_lam * entropy
        return loss, ll_mean.detach(), entropy.detach()

    # Entropy Dirichlet
    if exp_type == 5:
        entropy = Dirichlet(alpha).entropy().mean()
        loss = ll_mean - entropy_lam * entropy
        return loss, ll_mean.detach(), entropy.detach()
    
    if exp_type == 6:
        ce = nll(alpha, target)
        entropy = Dirichlet(alpha).entropy().mean()
        loss = ce - entropy_lam * entropy
        return loss, ce.detach(), entropy.detach()
    
    if exp_type == 7:
        entropy = Dirichlet(alpha).entropy().mean()
        ce = nll(alpha, target)
        loss = ll_mean + ce_lam*ce - entropy_lam * entropy
        return loss, ll_mean.detach(), ce.detach(), entropy.detach()
### ### 


def KL_expectedProbSL_teacherProb(alpha, pretrainedProb, forward=True):
    S = torch.sum(alpha, dim=1, keepdims=True)
    prob = alpha / S
    if forward:
        kl = kl_loss(torch.log(prob), pretrainedProb)
    else:
        kl = kl_loss(torch.log(pretrainedProb), prob)
    return kl


def nll(alpha, p_target):
    S = torch.sum(alpha, dim=1, keepdims=True)
    prob = alpha / S
    ce = F.nll_loss(torch.log(prob), p_target.max(1)[1])
    return ce 


def entropy_SL(alpha):
    S = torch.sum(alpha, dim=1, keepdims=True)
    prob = alpha / S
    entropy = - prob * torch.log(prob)
    entropy_s = torch.sum(entropy, dim=1)
    entropy_m = torch.mean(entropy_s)
    return entropy_m


## the followings are for HENN mainly 
def squaredLoss(alpha, one_hot_vectors, device):
    alpha_sum = torch.sum(alpha, dim=1)
    p_exp = torch.div(alpha, alpha_sum[:, None])

    num_classes = alpha.size(dim=1)
    num_samples = len(alpha)

    losses_term_1 = (one_hot_vectors - p_exp)**2
    losses_term_2 = (p_exp * (torch.ones(num_samples, num_classes, device=device) - p_exp) / ((alpha_sum + 1.0)[:, None]))

    losses = torch.sum(losses_term_1 + losses_term_2, dim=1)

    return losses


# did not use this 
def KL(alpha, device):
    num_classes = alpha.size(dim=1)
    beta = torch.ones(1, num_classes, device=device)
    S_beta = torch.sum(beta)
    S_alpha = torch.sum(alpha, dim=1) 
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha),dim=1)
    lnB_uni = torch.sum(torch.lgamma(beta),dim=1) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta)*(dg1-dg0[:, None]),dim=1) + lnB + lnB_uni

    return kl


def lossFunc(r, one_hot_labels, base_rate, num_single, annealing_coefficient):
    alpha = torch.add(r, torch.mul(num_single, base_rate)) # num_single:W
    squared_loss = squaredLoss(alpha, one_hot_labels)
    # kl = KL(alpha)
    # return torch.mean(squared_loss + (annealing_coefficient * kl))
    # return squared_loss + (annealing_coefficient * kl), squared_loss, kl
    return torch.mean(squared_loss)