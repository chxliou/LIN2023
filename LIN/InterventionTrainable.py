import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import time


def Cartesian_apply(*tensors, func):
    r"""
    Apply `fun` over Cartesian products among given tensors

    Output:
        a tensor `res`, such that
            res[i][j] = func(x1[i], x2[j])

            res[i][j][k] = func(x1[i], x2[j], x3[k])

    Args:
        *tensors: x_1, x_2, x_3, ..., x_n
        func: (n) -> out_shape
    
    Shape:
        res: (len(x1), len(x_2), ..., len(x_n), out_shape)
    
    Examples::
        >>> my_operation = lambda x,y: torch.tensor([x, y, x+y, x-y]).reshape(2,2)
        >>> res = Cartesian_apply(torch.arange(3),torch.arange(3), func = my_operation)
        >>> print(res.shape)
        torch.Size([3, 3, 2, 2])
    """

    args = torch.cartesian_prod(*tensors)
    res = [func(*arg) for arg in args] # 此步耗时最多
    out_shape = list(res[0].shape)
    res = torch.concat(res)
    shapes = [len(tensor) for tensor in tensors] + out_shape
    
    return res.reshape(*shapes)


def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)

def gumbel_sigmoid(log_alpha, uniform, tau=1, hard=True):
    device = log_alpha.device
    y_soft = torch.sigmoid((log_alpha + sample_logistic(log_alpha.shape,uniform).to(device))/ tau)
    

    if hard:
        y_hard = (y_soft > 0.5).type(torch.Tensor).to(device)

        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.clone().detach() - y_soft.clone().detach() + y_soft

    else:
        y = y_soft

    return y


class Interventions(nn.Module):
    r"""
    Args:
        d: number of variables.
        P: size of time range. e.g. `P=1` means no time lag.
        K: number of possible latent interventions.
        hidden_dim: size of hidden units.
        n_lyr: number of hidden layer.
        neg_slope: LeakyReLU's neg slope.
    
    Attributes:
        n_hidden_lyr, in_dim, hidden_dim, out_dim: networks' dimension
        nets: list of neuraul network for each intervention
            nets[intv_id][x_id]: (batch_size, d*P) -> (batch_size, n_param)
    """
    def __init__(self, d, P, K, hidden_dim = 5, out_dim = 1, shared_net_dim = 1, n_hidden_lyr = 1, neg_slope = 0.2, device = 'cuda:0', network_bias = True):
        # validation
        assert n_hidden_lyr >= 1
        self.device = device

        # init basic
        super(Interventions, self).__init__()
        self.d, self.P, self.K = d, P, K 

        # init network dimensions
        self.n_hidden_lyr = n_hidden_lyr
        self.in_dim = self.d * self.P
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.shared_net_dim = shared_net_dim
        self.uniform = torch.distributions.uniform.Uniform(1e-2,1-1e-2)

        # init target log_prob
        self.targets = Parameter(torch.zeros((d, self.K, self.K-1)).to(self.device))
        self.targets_mask = torch.tril(torch.ones((d, self.K, self.K-1)), diagonal = -1).to(self.device)
        #self.targets_mask = torch.tril(torch.ones((d, self.K, self.K)), diagonal = 0).to(self.device)

        # init network 
        # self.nets_fun = {}
        self.nets = nn.ModuleDict()
        for intv_id in range(self.K):
            intv_id = str(intv_id)
            # self.nets_fun[intv_id] = {}
            self.nets[intv_id] = nn.ModuleDict()
            for x_id in range(self.d):
                x_id = str(x_id)
                self.nets[intv_id][x_id] = nn.Sequential()
                self.nets[intv_id][x_id].append(nn.Linear(self.in_dim, self.hidden_dim, bias = network_bias).to(device))
                self.nets[intv_id][x_id][-1].reset_parameters()
                self.nets[intv_id][x_id].append(nn.LeakyReLU(negative_slope=neg_slope).to(device))

                for _ in range(self.n_hidden_lyr - 1):
                    self.nets[intv_id][x_id].append(nn.Linear(self.hidden_dim, self.hidden_dim, bias = network_bias).to(device))
                    self.nets[intv_id][x_id][-1].reset_parameters()
                    self.nets[intv_id][x_id].append(nn.LeakyReLU(negative_slope=neg_slope).to(device))
                
                self.nets[intv_id][x_id].append(nn.Linear(self.hidden_dim + self.in_dim, self.out_dim, bias = network_bias).to(device))
                self.nets[intv_id][x_id][-1].reset_parameters()
                self.nets[intv_id][x_id].append(nn.LeakyReLU(negative_slope=neg_slope).to(device))
            
        
        # init std network 
        # self.std_nets_fun = {}
        self.std_nets = nn.ModuleDict()
        for intv_id in range(self.K):
            intv_id = str(intv_id)
            # self.std_nets_fun[intv_id] = {}
            self.std_nets[intv_id] = nn.ModuleDict()
            for x_id in range(self.d):
                x_id = str(x_id)
                self.std_nets[intv_id][x_id] = nn.Sequential()
                self.std_nets[intv_id][x_id].append(nn.Linear(self.in_dim, self.hidden_dim, bias = network_bias).to(device))
                self.std_nets[intv_id][x_id][-1].reset_parameters()
                self.std_nets[intv_id][x_id].append(nn.LeakyReLU(negative_slope=neg_slope).to(device))

                for _ in range(self.n_hidden_lyr - 1):
                    self.std_nets[intv_id][x_id].append(nn.Linear(self.hidden_dim, self.hidden_dim, bias = network_bias).to(device))
                    self.std_nets[intv_id][x_id][-1].reset_parameters()
                    self.std_nets[intv_id][x_id].append(nn.LeakyReLU(negative_slope=neg_slope).to(device))
                
                self.std_nets[intv_id][x_id].append(nn.Linear(self.hidden_dim + self.in_dim, self.out_dim, bias = network_bias).to(device))
                self.std_nets[intv_id][x_id][-1].reset_parameters()
                self.std_nets[intv_id][x_id].append(nn.LeakyReLU(negative_slope=neg_slope).to(device))
        
        
    
    @torch.no_grad()
    def update_params(self, cut_bd):
        # validation
        assert cut_bd > 0

        self.targets.data = torch.clamp(self.targets.data, -cut_bd, cut_bd)

        
    def get_tgt_pnt(self):
        r"""
        Output
            penalty for targets.

        Shape
            - Output: scalar
        """
        # validattion
        assert self.K > 1

        # calc
        target_resolved = torch.concat([torch.zeros((self.d,self.K,1)).to(self.device),(self.targets * self.targets_mask)],2) + torch.diag(torch.ones(self.K)).to(self.device)
        #target_resolved = (self.targets * self.targets_mask)
        for j in range(self.d):
            for k in range(self.K):
                target_resolved[j,k,:k+1] = F.softmax(target_resolved[j,k,:k+1], dim = 0)
                if k == 0:
                    target_resolved[j,k,:k+1] = target_resolved[j,k,:k+1].detach()

        pnt = (target_resolved.reshape(-1, self.K) @ torch.arange(self.K).to(self.device).float()).mean()

        return pnt

    def get_nll_node(self, x_values, x_inputs):
        # init
        params, shared_params= self.get_dist_params(x_inputs) #Shape (batch_size, d, n_interv, n_params)
        nll_loss = nn.GaussianNLLLoss(reduction = 'none')

        # calc funcs
        # 1. calc_nll_node (batch_id, x_id = i, intv_id = k) -> nll(x_i) under k-th intervention over batch
        # 2. target_mask (intv_id = k) -> [ prob{i \in I_k} for each node_id i ]
        #calc_nll_node = nll_loss(x_values.unsqueeze(2), params[:, :, :, 0].squeeze(),10.0 * torch.sigmoid(shared_params[:, :, 0]).unsqueeze(2))
        calc_nll_node = nll_loss(torch.concat([x_values.unsqueeze(2)] * self.K, 2), params[:, :, :, 0],(10.0 * torch.sigmoid(shared_params[:, :, :, 0])) ** 2)
        
        return calc_nll_node

    def get_nll(self, x_values, x_inputs):
        r'''
        Output:
            neg log likelihood to each intervention
        
        Args:
            x_values: target value to each conditional. 
            x_inputs: conditional variables related to each conditional. 

        Shape:
            - Input 1: (batch_size, d) 
            - Input 2: (batch_size, d, n_variables * P) 
            - Output: (batch_size, n_interv)
        '''
        calc_nll_node = self.get_nll_node(x_values, x_inputs) # shape: (n, d, E)
        
        #sftmx_prob = lambda y: F.softmax(torch.concat([y.reshape(-1,1), torch.zeros(self.d).to(self.device).reshape(-1,1)],1), dim = 1)
        sftmx_prob = lambda y: torch.concat([torch.sigmoid(y).reshape(-1,1), (1-torch.sigmoid(y)).reshape(-1,1)],1)

        target_resolved = torch.concat([torch.zeros((self.d,self.K,1)).to(self.device),(self.targets * self.targets_mask)],2)  + torch.diag(torch.ones(self.K)).to(self.device)
        #target_resolved = (self.targets * self.targets_mask)
        for j in range(self.d):
            for k in range(self.K):
                target_resolved[j,k,:k+1] = F.softmax(target_resolved[j,k,:k+1], dim = 0)
                if k == 0:
                    target_resolved[j,k,:k+1] = target_resolved[j,k,:k+1].detach()

        nll = torch.concat([((calc_nll_node[..., :i+1] * target_resolved[:,i,:i+1])).sum(2).sum(1).unsqueeze(0) for i in range(self.K)]).T
        #nll = torch.concat([(calc_nll_node[:, :, [i, 0]] * sftmx_prob(self.targets[str(i)])).sum(2).sum(1).unsqueeze(0) for i in range(self.K)]).T
       
        return nll

    def get_dist_params(self, x_inputs):
        '''
        Output:
            (estimated) distribution parameters for each intervention
        
        Args:
            x_inputs: variables related ot each conditional. 

        Shape:
            - Input: (batch_size, d, n_variables * P)
            - Output: (batch_size, d, n_interv, n_params)
        '''

        # shape check: 
        #   1) output of nets[][]: (batch_size, n_param)
        #   2) .unsqueeze(1): (batch_size, 1, n_param)
        #   3) concat: (batch_size, n_interv, n_param)
        #   4) .unsqueeze(1): (batch_size, 1, n_interv, n_param)
        #   5) concat: (batch_size, d, n_interv, n_param)

        #concat_over_intv_id = lambda x_id: torch.concat([self.nets_fun[str(intv_id)][str(x_id)](x_inputs[:,x_id,:]).unsqueeze(1) for intv_id in range(self.K)], dim=1)
        concat_over_intv_id = lambda x_id: torch.concat([
            self.nets[str(intv_id)][str(x_id)][-2](torch.concat([self.nets[str(intv_id)][str(x_id)][:-2](x_inputs[:,x_id,:]), x_inputs[:,x_id,:]],1)).unsqueeze(1) for intv_id in range(self.K)], dim=1)
        all_params = torch.concat([ concat_over_intv_id(x_id).unsqueeze(1) for x_id in range(self.d) ], dim=1)

        #shared_params_per_node = lambda x_id: self.shared_nets[str(x_id)](x_inputs[:,x_id,:]) 
        #shared_params_per_node = lambda x_id: torch.concat([self.std_nets_fun[str(intv_id)][str(x_id)](x_inputs[:,x_id,:]).unsqueeze(1) for intv_id in range(self.K)], dim=1)
        shared_params_per_node = lambda x_id: torch.concat([
            self.std_nets[str(intv_id)][str(x_id)][-2](torch.concat([self.std_nets[str(intv_id)][str(x_id)][:-2](x_inputs[:,x_id,:]), x_inputs[:,x_id,:]],1)).unsqueeze(1) for intv_id in range(self.K)], dim=1)
        shared_params = torch.concat([ shared_params_per_node(x_id).unsqueeze(1) for x_id in range(self.d) ], dim=1)

        return all_params, shared_params
