'''
Usage:
    cd /root/autodl-tmp/LIN2023/Experiments_Results/Brownian

    python Brownian_LIN.py --dim 1 --graphtype ER --log --pnt 1e-8 --E 2
    python Brownian_LIN.py --dim 1 --graphtype ER --log --pnt 1e-8 --E 3
    python Brownian_LIN.py --dim 1 --graphtype ER --log --pnt 1e-8 --E 4
    python Brownian_LIN.py --dim 1 --graphtype ER --log --pnt 1e-8 --E 5
    python Brownian_LIN.py --dim 1 --graphtype ER --log --pnt 1e-4 --E 2
    python Brownian_LIN.py --dim 1 --graphtype ER --log --pnt 1e-4 --E 3
    python Brownian_LIN.py --dim 1 --graphtype ER --log --pnt 1e-4 --E 4
    python Brownian_LIN.py --dim 1 --graphtype ER --log --pnt 1e-4 --E 5

    python Brownian_LIN.py --dim 2 --graphtype ER --log --pnt 1e-8 --E 2
    python Brownian_LIN.py --dim 2 --graphtype ER --log --pnt 1e-8 --E 3
    python Brownian_LIN.py --dim 2 --graphtype ER --log --pnt 1e-8 --E 4
    python Brownian_LIN.py --dim 2 --graphtype ER --log --pnt 1e-8 --E 5
    python Brownian_LIN.py --dim 2 --graphtype ER --log --pnt 1e-4 --E 2
    python Brownian_LIN.py --dim 2 --graphtype ER --log --pnt 1e-4 --E 3
    python Brownian_LIN.py --dim 2 --graphtype ER --log --pnt 1e-4 --E 4
    python Brownian_LIN.py --dim 2 --graphtype ER --log --pnt 1e-4 --E 5

    python Brownian_LIN.py --dim 1 --graphtype BA --log --pnt 1e-8 --E 2
    python Brownian_LIN.py --dim 1 --graphtype BA --log --pnt 1e-8 --E 3
    python Brownian_LIN.py --dim 1 --graphtype BA --log --pnt 1e-8 --E 4
    python Brownian_LIN.py --dim 1 --graphtype BA --log --pnt 1e-8 --E 5
    python Brownian_LIN.py --dim 1 --graphtype BA --log --pnt 1e-4 --E 2
    python Brownian_LIN.py --dim 1 --graphtype BA --log --pnt 1e-4 --E 3
    python Brownian_LIN.py --dim 1 --graphtype BA --log --pnt 1e-4 --E 4
    python Brownian_LIN.py --dim 1 --graphtype BA --log --pnt 1e-4 --E 5

    python Brownian_LIN.py --dim 2 --graphtype BA --log --pnt 1e-8 --E 2
    python Brownian_LIN.py --dim 2 --graphtype BA --log --pnt 1e-8 --E 3
    python Brownian_LIN.py --dim 2 --graphtype BA --log --pnt 1e-8 --E 4
    python Brownian_LIN.py --dim 2 --graphtype BA --log --pnt 1e-8 --E 5
    python Brownian_LIN.py --dim 2 --graphtype BA --log --pnt 1e-4 --E 2
    python Brownian_LIN.py --dim 2 --graphtype BA --log --pnt 1e-4 --E 3
    python Brownian_LIN.py --dim 2 --graphtype BA --log --pnt 1e-4 --E 4
    python Brownian_LIN.py --dim 2 --graphtype BA --log --pnt 1e-4 --E 5
'''
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import random

import sys
import os
sys.path.append('/root/autodl-tmp/LIN2023')
sys.path.append('/root/autodl-tmp/LIN2023/Experiments_Results')

from LIN.models.LIN_GaussianNet import LIN_GaussianNet
from LIN.utils import get_ri, get_ari, causalGraph_metrics, getData, Logger

# get input options
parser = argparse.ArgumentParser()
parser.add_argument('--E', type=int, default=3)
parser.add_argument('--dim', type=int, default=1)
parser.add_argument('--pnt', type=float, default=1e-8)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--graphtype', type=str, default='ER', help='ER or BA')
parser.add_argument('--uneven', action="store_true")
parser.add_argument('--log', action="store_true")
parser.add_argument('--P', type=int, default=2)
parser.add_argument('--rdseeds', type=int, default=0)
opt = parser.parse_args()

# REPRODUCIBILITY
torch.manual_seed(opt.rdseeds)
torch.cuda.manual_seed(opt.rdseeds)
np.random.seed(opt.rdseeds)
random.seed(opt.rdseeds)

# logging
file_save_path = f'/root/autodl-tmp/LIN2023/Experiments_Results/Brownian/lin_result'
file_name = f"Brownian_LIN_E{opt.E}_dim{opt.dim}_{opt.graphtype}" + '_' + str(int(-np.log10(opt.pnt)))
if not os.path.exists(file_save_path):
    os.makedirs(file_save_path)
if opt.log:
    log_path = os.path.join(file_save_path, file_name+'.log')
    print(log_path) 
    sys.stdout = Logger(log_path)

## Config
d = 5
K = 2 ** opt.dim
T = 5000 
P = opt.P

# load data
Data_save_path = '/root/autodl-tmp/LIN2023/Experiments_Results/Brownian/Brownian_data'
data_file_name = f"Brownian_{opt.dim}D_{opt.graphtype}"

G = torch.load(os.path.join(Data_save_path, data_file_name + '_G.pt'))
data = torch.load(os.path.join(Data_save_path, data_file_name + '_data.pt'))
intervention_type = torch.load(os.path.join(Data_save_path, data_file_name + '_intvs.pt'))

print(G)

# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
sys.stdout.flush()
data = data.to(device)
intervention_type = intervention_type.to(device)

def aug_vbs_fun(model, for_nni = False,  calc_dist = False):
    
    est_G = (torch.sigmoid(model.G).reshape(-1) > 0.5) * 1.0
    accu, recall, shd, sid, F1  = causalGraph_metrics(model.G, G, calc_dist= calc_dist, allow_skeletion= False)

    the_intv_p = model.intv_p
    ri = get_ri(intervention_type, the_intv_p.argmax(1))
    ari = get_ari(intervention_type, the_intv_p.argmax(1))

    print('\taccu\trecall\tF1\tshd\tsid\tri\tari')#accu, recall, tnr, rev_rate, F1
    print(('\t{:6.4f}' * 7).format(accu, recall, F1, shd, sid, ri, ari))

    print('\t', list(the_intv_p.argmax(1).unique(return_counts= True)[1].cpu().numpy()))

    print("\tclusters")
    for cluster_id in range(opt.E):
        print( ( '\t[' + ("{:.2f},\t"* K) + '],').format(*tuple((intervention_type[the_intv_p.argmax(1) == cluster_id]==domain).float().mean().item() for domain in range(K))) )


intv_args = {
    "hidden_dim": 5, 
    "n_hidden_lyr": 1 
}


fit_args = {
    "epoch": 1000, 
    "lr_net": 1e-2, 
    "lr": 1e-2, 
    "batch_size": 256,
    "train_sample": 0.8,
    "struct_pnt_coeff":  (opt.pnt, opt.pnt, 0), 
    "patient": 100,  
    "update_patient": 3, 
    "tol_rate": 0, 
    "aug_vbs_fun":aug_vbs_fun,
    "itr_per_epoch": 100,
    "sub_pb_pt": 1,
    "verbose_period": 10,
    "lag_delta": 0.99
}

assert not torch.any(torch.isnan(data))

    
# training
model = LIN_GaussianNet(d, P, opt.E, intv_args, best_model_path = os.path.join(file_save_path, file_name+'_model.pt'), device= device, lgr_init = 1e-8)
model = model.fit(data, **fit_args)

print('finish')
aug_vbs_fun(model, calc_dist = True)

print('empirical selection criterion (lower is better):', model._criteria)
print('log_path', log_path)
print('end')