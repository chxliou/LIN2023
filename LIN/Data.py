import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class Intervention():
    def __init__(self, targets, d, P, params = None, relu_neg_slope = 0.3, sigmoid_amplitude = 1, latent_dim = None):
        '''
        INPUT:
            targets: target of this intervention 
            d, P: usual meaning
            params: a dictionary to control generating process, strored as `self.params`. 
        '''
        assert max(targets) < d
        self.targets = np.array(list(set(targets)))

        self.tgt_init = 10 * (2 * torch.rand(1) - 1)
        print(self.targets, '\t', self.tgt_init)
        
        if params is not None:
            self.params = params
        else:
            self.params = {}
            self.tanh = lambda x : sigmoid_amplitude * nn.Tanh()(x)
            self.sigmoid = lambda x : sigmoid_amplitude * nn.Sigmoid()(x)
            self.relu = nn.LeakyReLU(relu_neg_slope)
            if latent_dim is None:
                latent_dim = [ max(5, d//2)]
                latent_dim = [P*d] + latent_dim + [1]

            for x_id in self.targets:
                params = {
                    'mu':[],
                    'sd':[]
                }

                for l in range(1, len(latent_dim)):
                    in_dim = latent_dim[l-1]
                    out_dim = latent_dim[l]

                    mu_lyer =  torch.randn(out_dim, in_dim).squeeze() 
                    sd_lyer =  torch.randn(out_dim, in_dim).squeeze() 

                    params['mu'].append( mu_lyer ) 
                    params['sd'].append( sd_lyer ) 
                self.params[x_id] = params
    
    def sample(self, x_id, x_input): 
        assert x_id in self.targets
        the_params = self.params[x_id]
        assert len(the_params['mu']) == len(the_params['sd'])
        L = len(the_params['sd'])

        if any(x_input) > 0:
            mu = x_input.clone()
            sd = x_input.clone()
            for l in range(L):

                mu = the_params['mu'][l] @ mu
                sd = the_params['sd'][l] @ sd            

                if l < L - 1:
                    mu = self.tanh(mu)
                    sd = self.tanh(sd)
                else:
                    #mu = self.relu(mu)
                    sd = self.sigmoid(sd)      
        else:
            mu = self.tgt_init #0.5 * (2 * torch.rand(1) - 1)
            sd = self.sigmoid(torch.tensor(0.0)) # i.e. sigmoid_amplitude / 2
        out = mu + sd * torch.randn(1).squeeze()

        # smoothing
        # if out > 1e8:
        #     out = 1e8 + torch.log(1 + out - 1e8)
        # if out <-1e8:
        #     out = -1e8 - torch.log(1 - out - 1e8)
        assert not torch.isnan(out)
        return out 

class TsData():
    def __init__(self, G, I, P = None, d = None) -> None:
        assert len(I)>0
        assert (P==None) | (P == G.shape[0])
        assert (d==None) | (d == G.shape[1])
        assert (G.shape[1] == G.shape[2])
        self.G, self.I, self.P, self.d = G, I, G.shape[0], G.shape[1]
    
    def generate(self, T, weight = None, given_intervention = None):
        '''
        INPUT:
            T: length of time series
        '''
        data = torch.randn(T, self.d)

        if given_intervention is None:
            if weight is None:
                weight = torch.ones(len(self.I)).float()
            intervention_type = torch.multinomial(weight, T, replacement = True)#np.zeros(T).astype(int)
        else:
            intervention_type = given_intervention.clone()

        def get_value(t,x_id):
            if not torch.isnan(data[t,x_id]):
                return data[t,x_id]
            # check dependency
            for i in range(self.d):
                if self.G[0][i,x_id] == 0:
                    continue
                if torch.isnan(data[t, i]):
                    data[t, i] = get_value(t,i)
            # build input_x
            input_x = torch.concat([torch.nan_to_num(data[t-p,:])*self.G[p][:,x_id]  for p in range(self.P)])
            if x_id in self.I[y_t].targets:
                return self.I[y_t].sample(x_id, input_x) 
            else:
                return self.I[0].sample(x_id, input_x)
        
        for t in tqdm(range(self.P-1, T), desc = 'generating data', total=T, initial = self.P-1):
            # init
            data[t,:] = torch.nan

            y_t = intervention_type[t]

            for x_id in range(self.d):
                if not torch.isnan(data[t,x_id]):
                    continue
                data[t,x_id] = get_value(t,x_id)

        return data, intervention_type

class Intervention_old():
    def __init__(self, targets, d, P, fixed_w_std = None, func = None) -> None:
        '''
        INPUT:
            targets[k]: x_id of kth target of this intervention 
            dists[k]: 
            paras[k]: 
        '''
        assert max(targets) < d
        self.targets = np.array(list(set(targets)))
        
        if func is None:
            func = lambda n: 0.5 + np.random.rand(n)

        self.params = []
        for _ in range(len(self.targets)):
            w = 2*np.random.rand(d*P) - 1
            w = (func(d*P)) * (w >= 0) - (func(d*P)) * (w < 0)
            if fixed_w_std is not None:
                w_std = fixed_w_std
            else:
                w_std = 2*np.random.rand(d*P) - 1
                w_std = w_std * 0.6 + 0.25 * (w_std >= 0) - 0.25 * (w_std < 0)
            param = {
                'mu': w,
                'mu_b': 0,#np.random.rand(),
                'sd': w_std[0],
                'sd_b': w_std[1]
            }
            self.params.append(param)        
    
    def sample(self, x_id, x_input): 
        assert x_id in self.targets
        # follow B.1 of DCDI paper
        k = int( np.where(self.targets==x_id)[0])
        this_param = self.params[k]
        x_input = torch.tensor(x_input, dtype = torch.float32)
        if any(x_input) > 0:
            mu = np.inner(x_input, this_param['mu']) + this_param['mu_b']
            sd = np.inner(x_input, this_param['sd']) 
            sd = min(sd, 10.0)
            sd = max(sd, -10.0)
            sd = 1/(1 + np.exp(-sd)) + this_param['sd_b']
        else:
            mu = 2
            sd = 1
        out = mu + sd * np.random.randn()
        return out 