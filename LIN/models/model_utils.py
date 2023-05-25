import torch
import numpy as np

def get_h(model):
    M = (torch.sigmoid(model.G[0]) >= 0.5) * 1.0 + torch.sigmoid(model.G[0]) - torch.sigmoid(model.G[0]).detach()
    h = torch.trace(torch.matrix_exp(M)) - model.d
    return h 

@torch.no_grad()
def update_lagrange(model, ita = 2, delta = 0.9, bd = 1e8): # 1e8

    # check if it is dag
    h = get_h(model)
    if torch.abs(h) < 1e-8:
        return
    
    # --> here means not dag

    # so we need to increase the weight on dag constrains
    last_h = model.last_h
    model.last_h = h.item()
    if np.abs(model.lagrange_gamma) < bd:
        model.lagrange_gamma += (model.lagrange_mu * h).cpu().numpy()
    if np.abs(model.lagrange_mu) < bd:
        if h > delta * last_h:
            model.lagrange_mu = model.lagrange_mu * ita


@torch.no_grad()
def update_parameters(model):
    model.G.data = torch.clamp(model.G.data, -model.G_cut_bd, model.G_cut_bd)
    for i in range(model.d):
        model.G.data[0][i][i] = -model.G_cut_bd
    model.Intvs.update_params(model.G_cut_bd)

def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)

def gumbel_sigmoid(log_alpha, uniform, tau=1, hard=True):
    y_soft = torch.sigmoid(log_alpha / tau)
    device = log_alpha.device

    if hard:
        y_hard = (y_soft > 0.5).type(torch.Tensor).to(device)

        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.clone().detach() - y_soft.clone().detach() + y_soft

    else:
        y = y_soft

    return y

def calc_KL_div(mu, sd):
    return torch.log(sd) + (1+mu ** 2) / (2 * sd ** 2) - 1/2