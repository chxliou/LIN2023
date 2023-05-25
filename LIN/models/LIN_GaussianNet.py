'''
this version is a modification from dev1114

once we roll back to best model, we need to keep training it without updating clustrs.
'''

from ..InterventionTrainable import Interventions, Cartesian_apply
from ..utils import get_ri, get_ari, causalGraph_metrics
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter 
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from .model_utils import *

class LIN_GaussianNet(nn.Module):
    def __init__(self, d, P, E, intv_args, verbose_period = 1, best_model_path = 'best_model.pth', device = 'cuda:0', lgr_init = 1e-8, normalize_h = False):
        super().__init__()
        # init
        self.d = d 
        self.P = P
        self.E = E

        # params
        self.G = Parameter(torch.rand(self.P, self.d, self.d))
        self.Intvs = Interventions(self.d, self.P, self.E, device = device, **intv_args)

        # config
        self.G_cut_bd = 5.0 
        self.verbose_period = verbose_period
        self.best_model_path = best_model_path

        self.lgr_init = lgr_init
        self.lagrange_gamma = 0
        self.lagrange_mu = self.lgr_init
        
        
        if normalize_h:
            self.h_normalizer = get_h(self).item()
        else:
            self.h_normalizer = 1.0

        self.last_h = get_h(self).item() / self.h_normalizer
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.device = device
        self.to(device)
       
    
    def load_model(self, K, path, intv_args = None):
        if not os.path.exists(path):
            print(f'no such file {path}')
            return False

        self.load_state_dict(torch.load(path))

    def fit(self, data, epoch = 20, lr_net = 1e-2, lr = 1e-2, batch_size = 256, train_sample = 0.8, struct_pnt_coeff = 1e-8, patient = 20, update_patient= 3, tol_rate = 0, aug_vbs_fun = None, itr_per_epoch = 100, cluster_updating_period = 1, ita = 2, sub_pb_pt = 1, verbose_period = 1, dag_check = 0):
        r'''
        Train the model
        Input:
            data: tensor with shape T, d
        '''
        self.train()
        T, d = data.shape
        assert d == self.d
        assert ita > 1
        assert self.best_model_path is not None
        assert dag_check >= 0

        # train & eval split
        spl_point = int(T * train_sample)
        self.train_period = torch.arange(self.P-1,spl_point)
        self.eval_period = torch.arange(spl_point, T)
        self.cluster_updating_period = cluster_updating_period

        self.optimizer = torch.optim.Adam([
            {'params':self.G ,'lr':lr},
            {'params':self.Intvs.nets.parameters() ,'lr':lr_net},
            {'params':self.Intvs.std_nets.parameters() ,'lr':lr_net},
            {'params':self.Intvs.targets ,'lr':lr_net}, 
        ])
        self.intv_p = self.manage_intv_p(data, None, 'init').to(self.device)
        self.update_parameters()

        self._param_ratio = sum(p.numel() for p in self.parameters() if p.requires_grad) / T

        best_loss = np.inf
        ptnt = 0
        eval_loss_cache = np.inf # for cluster control
        cnt_cache = 0
        rearange_line = self.last_h#10.0

        lg_ptt = 10
        rlx_bd = 5.0
        for epoch_idx in range(epoch):
        
            # train 
            all_loss, all_nll, _ = self.go_through_batch(data, self.train_period, batch_size, struct_pnt_coeff, "train", itr_per_ep = itr_per_epoch, sub_pb_pt = sub_pb_pt)
            self.intv_p[self.eval_period] = self.manage_intv_p(data, self.eval_period, 'update', self.intv_p[self.eval_period]).detach()

            # eval
            eval_loss, eval_nll, _ = self.go_through_batch(data, self.eval_period, len(self.eval_period), struct_pnt_coeff, "eval", itr_per_ep = itr_per_epoch)
            self.intv_p[self.train_period] = self.manage_intv_p(data, self.train_period, 'update', self.intv_p[self.train_period]).detach()

            # early stop 
            this_h = get_h(self) / self.h_normalizer
            is_DAG = 'No' if ( this_h > dag_check) else 'Yes' 
            if (is_DAG == 'Yes'):
                if best_loss > eval_loss + tol_rate:
                    best_loss = eval_loss
                    ptnt = 0
                    # update criteria
                    self._best_loss = best_loss
                    self._spacity_I = self.Intvs.get_tgt_pnt()
                    self._spacity_G = (torch.sigmoid(self.G) >= 0.5).float().mean()
                    self._criteria = self._param_ratio + self._best_loss
                    self.last_h = get_h(self).item() / self.h_normalizer
                    
                    # save model
                    torch.save(self, self.best_model_path)  

                else:
                    ptnt += 1
                    if ptnt == patient:
                        break
                
            # update
            if eval_loss_cache >  eval_loss + tol_rate:
                cnt_cache = 0
                eval_loss_cache = eval_loss
            else:
                cnt_cache += 1
                if cnt_cache >=  update_patient:
                    cnt_cache = update_patient + 1
                    if is_DAG == 'No':
                        update_lagrange(self)
                        pass
                    eval_loss_cache = np.inf          

            # verbose
            if epoch_idx % verbose_period == 0:
                handle_none = lambda x: 9 if x is None else x
                print('Epoch:{:03d}\n\tloss:{:6.4f}\teval loss:{:6.4f}\tBest loss:{:6.4f}\th:{:.10f}'.format(epoch_idx + 1, all_loss, eval_loss, handle_none(best_loss),this_h))
                print('\tnll:{:6.4f}\teval.nll:{:6.4f}'.format(all_nll, eval_nll))
                if aug_vbs_fun is not None:
                    aug_vbs_fun(self)
                sys.stdout.flush()

        if get_h(self) > 0:
            if best_loss == np.inf:
                print('[WARNING] Not a DAG while attained maximal number of epochs')

        return torch.load(self.best_model_path)
    
    def updating_targets_prob(self, data, period, clpb,  lr = .5):
        '''
        Using given (data, period) to updating `targets` for each intervention
        '''
        # prepare
        x_values, x_inputs = self.prepare_inputs(data, period)
        params, shared_params= self.Intvs.get_dist_params(x_inputs) #Shape (batch_size, d, n_interv, n_params)

        # (batch_size, d, E)
        nll_loss = nn.GaussianNLLLoss(reduction = 'none')
        calc_nll_node = nll_loss(torch.concat([x_values.unsqueeze(2)] * self.Intvs.K, 2), params[:, :, :, 0],(10.0 * torch.sigmoid(shared_params[:, :, :, 0])) ** 2)

        # updating. clpb (batch_size, E)
        for intv_id in range(1, self.E):
            # cluster prob
            nll_cmp_tb = (clpb[:,[0,intv_id]].unsqueeze(1) * calc_nll_node[...,[0,intv_id]]).sum(0) / clpb[:,[0,intv_id]].sum(0)
            
            # update
            new_one = F.softmax(-nll_cmp_tb, dim = 1) + 1e-3
            new_one /= new_one.sum(1).unsqueeze(1)
            new_one = new_one[:,1].detach().to(self.device)
            assert new_one.min() > 0

            self.Intvs.targets[str(intv_id)] = (1-lr) * self.Intvs.targets[str(intv_id)] + lr * new_one


    def go_through_batch(self, data, the_period, batch_size, struct_pnt_coeff,cmd = "train", optimizer = None, itr_per_ep = 5, sub_pb_pt = 3, with_constrain = True):
        # validation
        assert cmd in ["train", "eval"]

        if optimizer is None:
            optimizer = self.optimizer

        all_loss = 0
        all_nll = 0
        all_dag = 0
        cnt = 0

        best_loss = np.inf
        sub_patient = sub_pb_pt
        sub_p_ct = 0

        # go thorgh epoch
        for t in range(itr_per_ep):
            # init
            period = torch.tensor(np.random.choice(the_period, batch_size, True ))
            if len(period) < 1:
                continue
            
            # calc
            nll = self.forward(data, period)
            pnt, dag = self.get_structure_pnt(struct_pnt_coeff)
            dism_pnt = struct_pnt_coeff[2] * self.get_disim_pnt(data, period)
            if with_constrain:
                loss = nll + pnt + dag + dism_pnt
            else:
                loss = nll + pnt

            # record
            all_loss += loss.item()
            all_nll += nll.item()
            all_dag += dag.item()
            cnt +=1

            if cmd == "train":
                # back propagation & learn
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.update_parameters()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    sub_p_ct = 0
                else:
                    sub_p_ct += 1

                if sub_p_ct >= sub_patient:
                    break

            else:
                break
                          
        return all_loss/cnt, all_nll/cnt, all_dag/cnt
    
    def update_targets_param(self, data, period):
        T, d = data.shape
        with torch.no_grad():
            x_values, x_inputs = self.prepare_inputs(data, period)
            calc_nll_node = self.Intvs.get_nll_node(x_values, x_inputs)
        for intv_id in range(1, self.E):
            this_wt = self.intv_p[period ,intv_id].reshape(-1,1,1)
            this_tbl = calc_nll_node[...,[0,intv_id]]
            new_one = (this_wt * this_tbl).sum(0) / this_wt.sum() @ torch.tensor([1., -1.]).to(self.device).detach()
            self.Intvs.targets[str(intv_id)] =  new_one


    @torch.no_grad()
    def manage_intv_p(self, data, period, cmd, prev_intv_p = None):
        r"""
        Manege `self.intv_p` 

        Details:
            `self.intv_p[t, intv_id]`: prob that sample at time t is controled by intv_id-th intv.
        """
        # validation
        assert cmd in ['init', 'update', 'sample']
        
        # init
        T, d = data.shape
        
        # init case
        if cmd == 'init':
            intv_p = torch.rand(T, self.E).to(self.device)
            intv_p = (intv_p.T / intv_p.sum(1)).T
  
        # update case
        if cmd == 'update':
            # validation
            assert period is not None
            assert hasattr(self, 'intv_p')
            assert prev_intv_p is not None

            # get p_c. Shape: (len of period)
            p_c = prev_intv_p.mean(0) + torch.arange(self.E,0,-1).to(self.device) * 0.01 
            t0 = self.P - 1
    
            # get nll. Shape: (len of period, n_interv)
            x_values, x_inputs = self.prepare_inputs(data, period)
            nll_table = self.Intvs.get_nll(x_values, x_inputs)

            # clustering based on nll
            ll_table = torch.clamp(-nll_table, -20.0, 20.0) 
            intv_p = F.gumbel_softmax(ll_table, dim = 1, hard=True).to(self.device) 

            assert intv_p.isnan().sum() ==0
            assert intv_p.min() >=0
            assert intv_p.max() < torch.inf

        return intv_p

    def update_centriods(self, data, period):
        with torch.no_grad():
            x_values, x_inputs = self.prepare_inputs(data, period)
            x_tildes = self.Intvs.get_normalized_samples(x_values, x_inputs)
        intv_p = self.intv_p[self.train_period]
        centriods = []
        centriods_invcov = []
        for intv_id in range(self.E):
            this_prob = intv_p[:,intv_id].reshape(-1,1)
            this_centroid = (x_tildes[...,intv_id] * this_prob).sum(0) / this_prob.sum()
            this_cov_list = torch.concat([ torch.outer(v-this_centroid,v-this_centroid).unsqueeze(0) for v in x_tildes[...,intv_id]])
            this_cov = (this_prob.unsqueeze(2) * this_cov_list).sum(0) / this_prob.sum()
            centriods.append(this_centroid.unsqueeze(0))
            centriods_invcov.append(this_cov.inverse().unsqueeze(0))
        return torch.concat(centriods), torch.concat(centriods_invcov)

    def get_disim_pnt(self, data, period):
        
        pnt = 0
    
        return pnt / ((self.E-1) * self.E /2)

    def forward(self, data, period):
        r'''
        Output:
            loss = \avg_k mean_nll(k)
            mean_nll(k): the mean neg log likelihood of k-th `cluster` over a time period (mini-batch)

        Args:
            data: time series data
            period: list of time index
        
        Shape:
            - Input 1: (T, d)
            - Input 2: (batch_size)
            - Output: Scalar
        
        Related Variabls:
            self.intv_p[t, intv_id]: prob that sample at time t is controled by intv_id-th intv.
        '''
        # init
        x_values, x_inputs = self.prepare_inputs(data, period)

        # get nll. Shape: (batch_size, n_interv)
        nll_table = self.Intvs.get_nll(x_values, x_inputs)

        # get intv_prob: (batch_size, n_interv)
        intv_prob = self.intv_p[period, :]
        
        loss = (intv_prob * nll_table).sum(1).mean() #.mean()
        return loss
    
    def prepare_inputs(self, data, period):
        r"""
        Outputs:
            x_values: samples to each conditional in period
            x_inputs: related variables to each conditional in period
        
        Args:
            data: time series data
            period: list of time index
        
        Shape:
            - Input 1: (T, d)
            - Input 2: (batch_size)
            - Output 1: (batch_size, d)
            - Output 2: (batch_size, d, n_variables * P)
        """
        # validation
        assert period.min() >= self.P - 1

        # prepare x_values. Shape: (batch_size, d) 
        x_values = data[period, :]

        with torch.no_grad():
            for i in range(self.d):
                self.G[0][i,i] = -self.G_cut_bd
        
        sts = period[0]
        T_len = len(period)
        x_inputs = torch.concat([
           (gumbel_sigmoid(self.G[p][:, :], self.uniform).unsqueeze(0) * data[period - p,:].unsqueeze(2)) for p in range(self.P)
        ],1).transpose(2,1)

        return x_values, x_inputs
    
    
    def get_structure_pnt(self, pnt_coeff):
        ''' 
        hope G is sparse,
        '''

        # pnt_G
        pnt_G = 0
        for p in range(self.P):
            for i in range(self.d):
                for j in range(self.d):
                    if (p==0) & (i==j):
                        continue
                    # note that the range of below is clamped to -5 to 5.
                    # this change should make gradient more clear
                    pnt_G += torch.sigmoid(self.G[p][i][j]) /(self.P * self.d ** 2) # self.G[p][i][j] /(self.P * self.d ** 2)#
        
        # pnt_I
        pnt_I = self.Intvs.get_tgt_pnt()

        # pnt_dag
        h = get_h(self) / self.h_normalizer
        dag = self.lagrange_gamma * h + self.lagrange_mu / 2 * h**2

        return pnt_coeff[0]*pnt_G + pnt_coeff[1]*pnt_I, dag

    @torch.no_grad()
    def update_parameters(self):
        self.G.data = torch.clamp(self.G.data, -self.G_cut_bd, self.G_cut_bd)
        for i in range(self.d):
            self.G.data[0][i][i] = -self.G_cut_bd
        self.Intvs.update_params(self.G_cut_bd)
        self.intv_p.data = torch.clamp(self.intv_p.data, -self.G_cut_bd, self.G_cut_bd)
