from .InterventionTrainable import Interventions, Cartesian_apply
from .utils import get_ri, get_ari, causalGraph_metrics
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
#import nni

class GaussianNet(nn.Module):
    def __init__(self, d, P, verbose_period = 1, tru_G = None, latent_intervention = None, best_model_path = 'best_model.pth', device = 'cpu', lgr_int = 1e-8) -> None:
        super().__init__()
        # init
        self.d = d 
        self.P = P 
        self.reset_all()

        # config
        self.verbose_show_graph = True
        self.G_cut_bd = 10.0
        self.lagrange_update_period = 1
        self.verbose_period = verbose_period
        self.tru_G = None if tru_G is None else torch.tensor(tru_G).float().to(device)
        self.latent_intervention = None if latent_intervention is None else latent_intervention.clone().to(device)
        self.best_model_path = best_model_path
        self.device = device
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.intv_args = None

    def reset_all(self):
        self.G = Parameter(torch.rand(self.P, self.d, self.d))
        self.started = False
        self.lagrange_gamma = 0
        self.lagrange_mu = 1e-8
    
    def start_model(self, K, device = None, cmd = None, intv_args = None):
        assert K >= 1

        if cmd == 'restart':
            self.reset_all()
        
        if device is None:
            device = self.device

        assert (self.started == False ), 'model has been started. set cmd="restart" to erace previous parameters'
        
        self.K = K
        if self.intv_args is None:
            self.intv_args = {}
        if intv_args is not None:
            self.intv_args = intv_args
        self.Intvs = Interventions(self.d, self.P, self.K, device = device, **self.intv_args)
        self.to(device)
        self.started = True    
        #self.intv_p = self.manage_intv_p(data, None, 'init').to(device)
    
    def load_model(self, K, path, intv_args = None):
        if not os.path.exists(path):
            print(f'no such file {path}')
            return False

        self.start_model(K, intv_args = intv_args)
        self.load_state_dict(torch.load(path))
        #self.intv_p = self.manage_intv_p(data, None, 'init').to(self.device)

        #print('load successfully. run "model.reset_all()" before fit')

    def fit(self, data, E = 3, epoch = 20, lr = 5e-1, batch_size = 64, train_sample = 0.8, struct_pnt_coeff = 0.01, max_epoch = 30, patient = 5, update_patient= 2, tol_rate = 0.01, bayes = False, cmd = None, for_nni = False, intv_args = None):
        r'''
        Train the model
        Input:
            data: tensor with shape T, d
            K: number of possible latent intervention, K>=1
        '''
        self.train()
        self.Bayes = bayes
        T, d = data.shape
        assert d == self.d

        
        # train & eval split
        spl_point = int(T * train_sample)
        self.train_period = torch.arange(self.P-1,spl_point, batch_size)
        self.eval_period = torch.arange(spl_point, T, batch_size)
        self.intv_args = intv_args

        # set init value
        self.last_h = self.get_h().item()

        if not self.started:
            self.start_model(E)
            print(f'current num of trainable parameter: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')
            self.intv_p = self.manage_intv_p(data, None, 'init').to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        best_loss = None
        ptnt = 0
        epoch_idx = 0
        eval_loss_cash = np.inf # for cluster control
        cnt_cash = 0
        while epoch_idx < epoch:

            # train
            all_loss, all_nll = self.go_through_batch(data, self.train_period, batch_size, struct_pnt_coeff, "train")

            # check DAG
            this_h = self.get_h()
            is_DAG = 'No' if ( this_h > 0) else 'Yes'   

            # eval 
            eval_loss, eval_nll = self.go_through_batch(data, self.eval_period, batch_size, struct_pnt_coeff, "eval")

            #self.update_lagrange()
            if (is_DAG == 'Yes'):
                if (best_loss is None) or (best_loss > eval_loss + tol_rate):
                    best_loss = eval_loss
                    ptnt = 0
                    torch.save(self.state_dict(), self.best_model_path)
                else:
                    ptnt += 1
                    if ptnt == patient:
                        break    
            
            if eval_loss_cash >  eval_loss + tol_rate:
                cnt_cash = 0
                eval_loss_cash = eval_loss
            else:
                cnt_cash += 1

            # update
            if (epoch_idx >= 10 ) and (cnt_cash >= update_patient):
                full_period = torch.arange(self.P - 1, T)
                self.intv_p[full_period] = self.manage_intv_p(data, full_period, 'update', self.intv_p[full_period])
                if is_DAG == 'No':
                    self.update_lagrange()
                #cnt_cash = 0
                eval_loss_cash = np.inf

            # verbose
            if epoch_idx % self.verbose_period == 0:
                handle_none = lambda x: 9 if x is None else x
                print('Epoch:{:03d}\n\tloss:{:6.4f}\teval loss:{:6.4f}\tBest loss:{:6.4f}\th:{:6.4f}'.format(epoch_idx + 1, all_loss, eval_loss, handle_none(best_loss),this_h))
                print('\t\tnll:{:6.4f}\teval.nll:{:6.4f}'.format(all_nll, eval_nll))
                if self.tru_G is not None:
                    est_G = (torch.sigmoid(self.G).reshape(-1) > 0.5) * 1.0
                    accu, recall, tnr, rev_rate, F1 = causalGraph_metrics(self.G, self.tru_G)

                    if for_nni:
                        nni.report_intermediate_result(F1.item())
                    
                    print('\taccu:{:6.4f}\trecall:{:6.4f}\ttnr:{:6.4f}\trev_rate:{:6.4f}\tF1:{:6.4f}'
                           .format(accu, recall, tnr, rev_rate, F1))
                if self.latent_intervention is not None:
                    ri = get_ri(self.latent_intervention, self.intv_p.argmax(1))
                    ari = get_ari(self.latent_intervention, self.intv_p.argmax(1))
                    print('\tri:{:6.4f}\tari:{:6.4f}'.format(ri, ari))
                print(list(self.intv_p.argmax(1).unique(return_counts= True)[1].cpu().numpy()))
                sys.stdout.flush()

            # stop?
            epoch_idx += 1
            if (cmd == 'train') and (epoch_idx == epoch) and (self.get_h() > 0):
                print(f'G[0] is not a dag after {epoch_idx} epochs.')
                if epoch < max_epoch:
                    epoch += 10
                    print('Try 10 more epochs')
                else:
                    print('[WARNING] Not a DAG while attained maximal number of epochs')

        self.load_state_dict(torch.load(self.best_model_path)) 
        full_period = torch.arange(self.P - 1, T)
        self.intv_p[full_period] = self.manage_intv_p(data, full_period, 'update', self.intv_p[full_period])
        return self.go_through_batch(data, self.eval_period, batch_size, struct_pnt_coeff, "eval")

    def go_through_batch(self, data, the_period, batch_size, struct_pnt_coeff, cmd = "train", optimizer = None):
        # validation
        assert cmd in ["train", "eval"]

        if optimizer is None:
            optimizer = self.optimizer

        all_loss = 0
        all_nll = 0
        cnt = 0
        # go thorgh epoch
        for t in the_period:
            # init
            period = torch.arange(t, min(t+batch_size, data.shape[0]))#the_period.max()))
            if len(period) < 1:
                continue

            #self.intv_p[period] = self.manage_intv_p(data, period, 'update', self.intv_p[period])
            
            # calc
            nll = self.forward(data, period)
            pnt, dag = self.get_structure_pnt(struct_pnt_coeff)
            loss = nll + pnt + dag

            # record
            all_loss += loss.item()
            all_nll += nll.item()
            cnt +=1

            if cmd == "train":
                # back propagation & learn
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.update_parameters()
        return all_loss/cnt, all_nll/cnt

    @torch.no_grad()
    def manage_intv_p(self, data, period, cmd, prev_intv_p = None, tau = 1):
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
            intv_p = torch.rand(T, self.K).to(self.device)
            intv_p = (intv_p.T / intv_p.sum(1)).T
  
        # update case
        if cmd == 'update':
            # validation
            assert period is not None
            assert hasattr(self, 'intv_p')
            assert prev_intv_p is not None

            # get p_c. Shape: (len of period)
            p_c = prev_intv_p.mean(0)
            intv_p = prev_intv_p.clone()
            t0 = self.P - 1
    
            # get nll. Shape: (len of period, n_interv)
            with torch.no_grad():
                x_values, x_inputs = self.prepare_inputs(data, period)
                nll_table = self.Intvs.get_nll(x_values, x_inputs)

            likelihood_table = torch.exp(torch.clamp( (-nll_table) , -20, 20))

            # update
            if self.Bayes:
                post_prob = likelihood_table * p_c.unsqueeze(0)
            else:
                post_prob = likelihood_table #* p_c.unsqueeze(0)
            intv_p = post_prob / post_prob.sum(1).unsqueeze(1)
            # intv_p = torch.zeros(prev_intv_p.shape).to(prev_intv_p.device)
            # for i in range(self.K):
            #     intv_p[nll_table.argmin(1)==i,i] = 1.0
            

        return intv_p

    def forward(self, data, period, cmd = None):
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
        batch_size = x_values.shape[0]

        # get nll. Shape: (batch_size, n_interv)
        nll_table = self.Intvs.get_nll(x_values, x_inputs)

        # get intv_prob: (batch_size, n_interv)
        intv_prob = self.intv_p[period, :].clone()

        # calc func
        # mean_nll: (intv_id) -> the mean neg log likelihood of k-th `cluster`
        mean_nll = lambda intv_id: (nll_table[:,intv_id] * intv_prob[:,intv_id]).sum() / intv_prob[:,intv_id].sum()

        # calc loss
        loss, loss_d = 0, 0
        for intv_id in range(self.K):
            if intv_prob[:,intv_id].sum() < 1:
                continue
            loss += mean_nll(intv_id)
            assert not torch.all(torch.isnan(loss))
            loss_d += 1
        loss /= loss_d

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

        # Prepare x_inputs. Shape: (batch_size, d, n_variables * P)
        # def x_input(time_id, x_id):
        #     x_input = torch.cat(
        #         [ data[time_id-p,:]*torch.sigmoid(self.G[p][:, x_id]) for p in range(self.P)]
        #     )
        #     with torch.no_grad():
        #         x_input[x_id] = 0.0
        #     if any(torch.isnan(x_input)):
        #         print(
        #             'G', any(torch.isnan(self.G.reshape(-1))),
        #             'data', any(torch.isnan(x_values.reshape(-1))), )
        #         assert not any(torch.isnan(x_input))
        #     return x_input
        # x_inputs = Cartesian_apply(period, torch.arange(0, self.d), func = x_input)

        with torch.no_grad():
            for i in range(self.d):
                self.G[0][i,i] = -self.G_cut_bd
        
        sts = period[0]
        T_len = len(period)
        x_inputs = torch.concat([
           #(torch.sigmoid(self.G[p][:, :]).unsqueeze(0) * data[sts-p:sts-p + T_len,:].unsqueeze(2)) for p in range(self.P)
           (gumbel_sigmoid(self.G[p][:, :], self.uniform).unsqueeze(0) * data[sts-p:sts-p + T_len,:].unsqueeze(2)) for p in range(self.P)
        ],1).transpose(2,1)

        return x_values, x_inputs
    
    def get_structure_pnt(self, pnt_coeff):
        ''' 
        hope G is sparse,
        '''
        assert self.started

        # pnt_G
        pnt_G = 0
        for p in range(self.P):
            for i in range(self.d):
                for j in range(self.d):
                    if (p==0) & (i==j):
                        continue
                    # note that the range of below is clamped to -5 to 5.
                    # this change should make gradient more clear
                    pnt_G += torch.sigmoid(self.G[p][i][j]) /(self.P * self.d ** 2)
        
        # pnt_I
        pnt_I = self.Intvs.get_tgt_pnt()

        # pnt_dag
        h = self.get_h()
        dag = self.lagrange_gamma * h + self.lagrange_mu / 2 * h**2

        return pnt_coeff*(pnt_G + pnt_I), dag

    def get_h(self):
        M = (torch.sigmoid(self.G[0]) >= 0.5) * 1.0 + torch.sigmoid(self.G[0])
        with torch.no_grad():
            M -= torch.sigmoid(self.G[0])
        h = torch.trace(torch.matrix_exp(M)) - self.d
        return h 
    
    @torch.no_grad()
    def update_parameters(self):
        assert self.started
        self.G.data = torch.clamp(self.G.data, -self.G_cut_bd, self.G_cut_bd)
        for i in range(self.d):
            self.G.data[0][i][i] = -self.G_cut_bd
        self.Intvs.update_params(self.G_cut_bd)

    @torch.no_grad()
    def update_lagrange(self, ita = 2, delta = 0.9, bd = 1e8):
        assert self.started

        # check if it is dag
        h = self.get_h()
        if torch.abs(h) < 1e-8:
            #self.lagrange_gamma = 0
            #self.lagrange_mu = 1e-8
            return
        
        # --> here means not dag

        # so we need to increase the weight on dag constrains
        last_h = self.last_h
        self.last_h = h.item()
        if np.abs(self.lagrange_gamma) < bd:
            self.lagrange_gamma += (self.lagrange_mu * h).cpu().numpy()
        if np.abs(self.lagrange_mu) < bd:
            if h > delta * last_h:
                self.lagrange_mu = self.lagrange_mu * ita

                
def smooth(x_now, x_next, scale = 5.0):
    r"""
    Output:
        smoothed value of `x_next` on `x_now`.
    
    Args:
        x_now: current state.
        x_next: next state.
        scale: the larger, the smoother.
    """
    assert scale > 1.0
    return (x_now * (scale - 1.0) + x_next)/scale

def writer_record(writer, idx, records):
    for name, value in records.items():
        if torch.is_tensor(value):
            value = value.item()
        writer.add_scalar(name, value, idx)

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
    



        
        

        


