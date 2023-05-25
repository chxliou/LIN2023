import numpy as np
import torch
import sys
from .Data import TsData, Intervention
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
import cdt
from cdt.metrics import SID, SHD

class Logger(object):
    r'''
    -----------------------------------
    Python通过重写sys.stdout将控制台日志重定向到文件
    https://blog.51cto.com/u_15127589/4531760
    '''
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def causalGraph_metrics_smry_g(est_G_raw, tru_G_raw_raw, summury_g = True, calc_dist = False, allow_skeletion = False):
    assert summury_g
    P, d, _ = est_G_raw.shape

    # make sure is tensor
    if torch.is_tensor(tru_G_raw_raw):
        tru_G_raw = tru_G_raw_raw.clone()
    else:
        tru_G_raw = torch.tensor(tru_G_raw_raw)

    # convert est_G_raw to adj matirx
    if est_G_raw.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        est_G = est_G_raw.float().cpu().clone()
    else:
        est_G = (torch.sigmoid(est_G_raw) > 0.5).float().cpu().clone()

    est_G_copy = est_G.clone()

    # convert them to summary graph, if needed
    est_G_ready = torch.any(est_G,0).float().cpu().clone().unsqueeze(0)
    tru_G_ready = torch.any(tru_G_raw,0).float().cpu().clone().unsqueeze(0)
    

    # treat `-1`` as skeletion detection 
    for i in range(d):
        for j in range(d):
            if (est_G_ready[0,i,j] != -1) or (est_G_ready[0,j,i] != -1):
                continue
            if (tru_G_ready[0,i,j] == 0) and (tru_G_ready[0,j,i] == 0):
                # false skeleton case, treat as an arbitrary directed eadge
                est_G_ready[0,i,j] = 1
                est_G_ready[0,j,i] = 0
            else:
                if allow_skeletion:
                    # correct skeleton case, treat as the true eadge
                    est_G_ready[0,i,j] = tru_G_ready[0,i,j]
                    est_G_ready[0,j,i] = tru_G_ready[0,j,i]
                else:
                    est_G_ready[0,i,j] = tru_G_ready[0,j,i]
                    est_G_ready[0,j,i] = tru_G_ready[0,i,j]


    est_G = est_G_ready.clone()#.reshape(-1)
    tru_G = tru_G_ready.clone()#.reshape(-1)

    # ---
    # locs = torch.where(est_G == -1)[0]
    # est_G[locs] = tru_G[locs]    

    # eval part
    cnt = len(tru_G.reshape(-1))
    total_edge_in_true_graph = sum(tru_G.reshape(-1))
    correct_detected_edge = sum(tru_G.reshape(-1) * est_G.reshape(-1))
    correct_detected_idpd = sum( (1-tru_G.reshape(-1)) * (1-est_G.reshape(-1)))
    total_detected_edge = sum(est_G.reshape(-1))

    prec = correct_detected_edge/total_detected_edge #(tn + tp) / (tn + fp + fn + tp) # 1.0 * sum(est_G == tru_G) / len(tru_G)
    accu = (correct_detected_edge+correct_detected_idpd)/cnt
    recall = correct_detected_edge/total_edge_in_true_graph #(tp)/(tp + fn + 1e-8) if tru_G.sum() > 0 else 1
    
    # calc F1 score 
    F1 = 2 * (prec * recall) / (prec + recall)

    # distances
    if calc_dist:
        
        shd = (est_G.int().numpy() ^ tru_G.int().numpy()).sum() # SHD(true_G_for_dis, est_G_for_dis)
        sid = -1 #SID(true_G_for_dis, est_G_for_dis)
    else:
        shd = -1.0
        sid = -1.0

    return accu, recall, shd, sid, F1

def causalGraph_metrics(est_G_raw, tru_G_raw_raw, summury_g = False, calc_dist = False, allow_skeletion = True):

    assert not summury_g

    P, d, _ = est_G_raw.shape

    # make sure is tensor
    if torch.is_tensor(tru_G_raw_raw):
        tru_G_raw = tru_G_raw_raw.clone()
    else:
        tru_G_raw = torch.tensor(tru_G_raw_raw)

    # convert est_G_raw to adj matirx
    if est_G_raw.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        est_G = est_G_raw.float().cpu().clone()
    else:
        est_G = (torch.sigmoid(est_G_raw) > 0.5).float().cpu().clone()

    # convert them to summary graph, if needed
    if summury_g:
        est_G_ready = torch.any(est_G,0).float().cpu().clone().unsqueeze(0)
        tru_G_ready = torch.any(tru_G_raw,0).float().cpu().clone().unsqueeze(0)
    else:
        est_G_ready = est_G.float().cpu().clone()
        tru_G_ready = tru_G_raw.detach().float().cpu().clone()

    # treat `-1`` as skeletion detection 
    for i in range(d):
        for j in range(d):
            if (est_G_ready[0,i,j] != -1) or (est_G_ready[0,j,i] != -1):
                continue
            if (tru_G_ready[0,i,j] == 0) and (tru_G_ready[0,j,i] == 0):
                # false skeleton case, treat as an arbitrary directed eadge
                est_G_ready[0,i,j] = 1
                est_G_ready[0,j,i] = 0
            else:
                if allow_skeletion:
                    # correct skeleton case, treat as the true eadge
                    est_G_ready[0,i,j] = tru_G_ready[0,i,j]
                    est_G_ready[0,j,i] = tru_G_ready[0,j,i]
                else:
                    est_G_ready[0,i,j] = tru_G_ready[0,j,i]
                    est_G_ready[0,j,i] = tru_G_ready[0,i,j]


    est_G = est_G_ready.clone()#.reshape(-1)
    tru_G = tru_G_ready.clone()#.reshape(-1)

    # ---
    # locs = torch.where(est_G == -1)[0]
    # est_G[locs] = tru_G[locs]    

    # eval part
    cnt = len(tru_G.reshape(-1))
    total_edge_in_true_graph = sum(tru_G.reshape(-1))
    correct_detected_edge = sum(tru_G.reshape(-1) * est_G.reshape(-1))
    correct_detected_idpd = sum( (1-tru_G.reshape(-1)) * (1-est_G.reshape(-1)))
    total_detected_edge = sum(est_G.reshape(-1))

    prec = correct_detected_edge/total_detected_edge 
    accu = (correct_detected_edge+correct_detected_idpd)/cnt
    recall = correct_detected_edge/total_edge_in_true_graph
    
    # calc F1 score 
    F1 = 2 * (prec * recall) / (prec + recall)

    # distances
    if calc_dist:
        est_G_for_dis = np.zeros((P*d,P*d))
        est_G_for_dis[:,:d] = est_G.reshape(d*P,-1).numpy()
        true_G_for_dis = np.zeros((P*d,P*d))
        true_G_for_dis[:,:d] = tru_G.reshape(d*P,-1).numpy()

        shd = SHD(true_G_for_dis, est_G_for_dis)
        sid = SID(true_G_for_dis, est_G_for_dis)
    else:
        shd = -1.0
        sid = -1.0

    return accu, recall, shd, sid, F1



def get_ri(labels_true, labels_pred):
    ri = rand_score(labels_true.cpu(), labels_pred.cpu())
    return ri

def get_ari(labels_true, labels_pred):
    ari = adjusted_rand_score(labels_true.cpu(), labels_pred.cpu())
    return ari

def getRandomGraph(P, d, e_den = 0.5, graph_type='ER'):
    assert graph_type in ['ER', 'BA']
    G = np.zeros([P,d,d])

    if graph_type == 'ER':
        for i in range(d):
            for j in range(i+1, d):
                if np.random.rand(1) < e_den:
                    G[0][i][j] = 1.0

        # permute G[0]
        idx = np.arange(d)
        np.random.shuffle(idx)
        G[0] = G[0][idx,:][:,idx]
        G = torch.from_numpy(G).int()
        assert torch.trace(torch.matrix_exp(G[0].float())) == d

        for p in range(1, P):
            for i in range(d):
                for j in range(d):
                    if np.random.rand(1) < e_den:
                        G[p][i][j] = 1.0
    
    if graph_type == 'BA':
        for i in range(1, d):
            k_in = np.array([G[0][:i,v].sum() for v in range(i)]) + 1.0
            pr = k_in/k_in.sum()
            for j in np.random.choice(range(i),max(1, int(i * e_den)),p=pr, replace = False):
                G[0][i][j] = 1.0

        # permute G[0]
        idx = np.arange(d)
        np.random.shuffle(idx)
        G[0] = G[0][idx,:][:,idx]
        G = torch.from_numpy(G).int()
        assert torch.trace(torch.matrix_exp(G[0].float())) == d
        
        for p in range(1, P):
            for i in range(d):
                k_in = np.array([G[:,:,v].sum() for v in range(d)]) + 1.0
                pr = k_in/k_in.sum()
                for j in np.random.choice(range(d),max(1, int(d * e_den)),p=pr, replace = False):
                    G[p][i][j] = 1.0

    return G

def getRandomIntervention(P, d, g = None, num_tgt = None):
    # g is the number of additional intervention 
    if g is None:
        g = torch.randint(1,d, size=(1,))[0]
    assert (g<d) and (g>=0)

    Intv_list = [Intervention(torch.arange(d), d, P)]

    for i in range(g):
        if num_tgt is None:
            this_targets_num = torch.randint(1,d, size=(1,))[0]
        else:
            this_targets_num = num_tgt
        this_targets_set = torch.randperm(d)[:this_targets_num]
        Intv_list.append(Intervention(this_targets_set, d, P))
    
    return len(Intv_list), Intv_list

def getData(P, d, num_additional_interv = None, e_den = 0.5, num_tgt = 2, graph_type = 'ER'):
    r"""
        Generate random causal graph and its data-generator

    Input:
        P: time-cross. 1 for instantaneous, ...
        d: number of nodes
        num_additional_interv: number of additional interventions. if not specified, random one.
    
    Output:
        G: Causal Graph
        K: number of total interventions
        Data: Data-Generators
        d, P: echos of input
    """
    
    G = getRandomGraph(P,d,e_den,graph_type)

    # Intv_list = [Intervention(np.arange(d), d, P, w_std, func_w)]
    # for i in range(3, d):
    #     if np.random.rand(1) > 0.5:
    #         Intv_list.append(Intervention(np.arange(i)[-3:], d, P, w_std, func_w))

    K, Intv_list = getRandomIntervention(P, d, num_additional_interv, num_tgt)
    
    Data = TsData(
        G = G,
        I = Intv_list
    )

    return G, K, Data