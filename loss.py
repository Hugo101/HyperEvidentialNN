import torch
import torch.nn.functional as F
from common_tools import set_device

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
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(
    y, alpha, epoch_num, num_classes, 
    annealing_step, kl_lam, 
    anneal=False, kl_reg=True, 
    device=None):
    if not device:
        device = set_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    if not kl_reg:
        return loglikelihood

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_term = kl_divergence(kl_alpha, num_classes, device=device)
    
    if anneal:
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )    
        kl_div = annealing_coef * kl_term
    else:
        kl_div = kl_lam * kl_term

    return loglikelihood + kl_div


def edl_loss(
    func, y, alpha, epoch_num, num_classes, 
    annealing_step, kl_lam, 
    anneal=False, kl_reg=True, 
    device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not kl_reg:
        return A 
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_term = kl_divergence(kl_alpha, num_classes, device=device)
    
    if anneal:
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )
        kl_div = annealing_coef * kl_term
    else:
        #todo 
        kl_div = kl_lam * kl_term

    return A + kl_div


def edl_mse_loss(
    output, target, epoch_num, num_classes, 
    annealing_step, kl_lam, 
    anneal=False, kl_reg=True, 
    device=None):
    if not device:
        device = set_device()
    # evidence = relu_evidence(output)
    evidence = output
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(
            target, alpha, epoch_num, num_classes, 
            annealing_step, kl_lam, 
            anneal=anneal, kl_reg=kl_reg, 
            device=device
        )
    )
    return loss

# todo:
def edl_log_loss(
    output, target, epoch_num, num_classes, 
    annealing_step, 
    kl_reg=True, 
    device=None):
    if not device:
        device = set_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, kl_reg=kl_reg, device=device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, 
    annealing_step, kl_lam, 
    anneal=False, kl_reg=True, 
    device=None
):
    if not device:
        device = set_device()
    # evidence = relu_evidence(output)
    evidence = output
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, 
            annealing_step, kl_lam, 
            anneal=anneal, kl_reg=kl_reg, 
            device=device
        )
    )
    return loss
### ### 



## the followings are for HENN mainly 
def squaredLoss(alpha, one_hot_vectors, device):
    alpha_sum = torch.sum(alpha, dim=1)
    p_exp = torch.div(alpha, alpha_sum[:, None])

    num_classes = alpha.size(dim=1)
    num_samples = len(alpha)

    losses_term_1 = (one_hot_vectors - p_exp)**2
    losses_term_2 = (p_exp * (torch.ones(num_samples, num_classes).to(device) - p_exp) / ((alpha_sum + 1.0)[:, None]))                   

    losses = torch.sum(losses_term_1 + losses_term_2, dim=1)

    return losses


# did not use this 
def KL(alpha, device):
    num_classes = alpha.size(dim=1)
    beta = torch.ones(1, num_classes).to(device)
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