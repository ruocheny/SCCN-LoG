import scipy
import torch
import math
import scipy.sparse as sp
import numpy as np


""" simplicial complex is built from cora and pubmed dataset where simlices are cliques
    simplicial complex is built from dblp coauthorship dataset (ref: hyperGCN)"""
def load_data_sc(dataname, modelname, pseudo_simplex):

    print('loading simplicial complex data from {}'.format(dataname))

    if pseudo_simplex:
        path = 'data/{}/pseudo/'.format(dataname)
    else:
        path = 'data/{}/'.format(dataname)

    mask_s = torch.load(path+'masks.pt'.format(dataname)) # mask for scn (simplicial convolutional networks)
    if modelname == 'scn' or modelname == 'gcn':
        s = torch.load(path+'input_s.pt'.format(dataname, dataname))
        return s['adj_norm_sp'], s['simplex_features'], s['simplex_labels'], s['simplex_chains'], \
            mask_s['idx_train'], mask_s['idx_val'], mask_s['idx_test']
    elif modelname == 'sccn' or modelname == 'mlp':
        sc = torch.load(path+'input_sc.pt'.format(dataname))
        sc_idx_train = torch.cat(mask_s['idx_train'],0)
        sc_idx_val = torch.cat(mask_s['idx_val'],0)
        sc_idx_test = torch.cat(mask_s['idx_test'],0)

        p_max = len(mask_s['idx_train']) - 1
        s_idx_train = torch.zeros(p_max+1,len(sc_idx_train),dtype=torch.bool)
        s_idx_val = torch.zeros(p_max+1,len(sc_idx_val),dtype=torch.bool)
        s_idx_test = torch.zeros(p_max+1,len(sc_idx_test),dtype=torch.bool)

        num_simplex = [len(mask_s['idx_train'][p]) for p in range(p_max+1)]

        for p in range(p_max+1):
            s_idx_train[p][sum(num_simplex[0:p]):sum(num_simplex[0:p+1])] = mask_s['idx_train'][p]
            s_idx_val[p][sum(num_simplex[0:p]):sum(num_simplex[0:p+1])] = mask_s['idx_val'][p]
            s_idx_test[p][sum(num_simplex[0:p]):sum(num_simplex[0:p+1])] = mask_s['idx_test'][p]
        

        return sc['adj'], sc['feat'], sc['label'], sc_idx_train, sc_idx_val, sc_idx_test, \
            s_idx_train, s_idx_val, s_idx_test, num_simplex





def gen_masks(dataname, modelname, pseudo_simplex):

    if pseudo_simplex:
        path = 'data/{}/pseudo/'.format(dataname)
    else:
        path = 'data/{}/'.format(dataname)
    
    s = torch.load(path + 'input_s.pt')
    simplex_labels = s['simplex_labels']
    

    """masks"""
    idx_train, idx_val, idx_test = [], [], []
    for i in range(len(simplex_labels)):
        sample_num = len(simplex_labels[i])

        if sample_num>0:
            train_ratio = min(300/sample_num, 0.6) # flexible train ratio according to different sample set
            val_ratio = min(300/sample_num, 0.2)

            test_ratio = 1 - train_ratio - val_ratio

            idx_train_i, idx_val_i, idx_test_i = train_test_split(sample_num, train_ratio, val_ratio, test_ratio)
        else:
            idx_train_i, idx_val_i, idx_test_i \
                = torch.tensor([],dtype=torch.bool), torch.tensor([],dtype=torch.bool), torch.tensor([],dtype=torch.bool)

        idx_train.append(idx_train_i)
        idx_val.append(idx_val_i)
        idx_test.append(idx_test_i)
    

    torch.save({'idx_train':idx_train, 'idx_val':idx_val, 'idx_test':idx_test},path+'masks.pt'.format(dataname))




def train_test_split(sample_num, train_ratio, val_ratio, test_ratio):
    permu = np.random.permutation(sample_num)
    train_num = int(sample_num * train_ratio)
    val_num = int(sample_num * val_ratio)
    test_num = int(sample_num * test_ratio)
    idx_train = torch.zeros(sample_num,dtype=torch.bool)
    idx_val = torch.zeros(sample_num,dtype=torch.bool)
    idx_test = torch.zeros(sample_num,dtype=torch.bool)
    idx_train[permu[:train_num]] = 1
    idx_val[permu[train_num:train_num+val_num]] = 1
    idx_test[permu[train_num+val_num:]] = 1
    return idx_train, idx_val, idx_test


"""this part of codes are from pygcn"""
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

