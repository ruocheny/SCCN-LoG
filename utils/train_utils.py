import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    
    return correct / len(labels)

def loss_each(output, labels, idx, threshold):
    loss_each = F.nll_loss(output[idx], labels[idx], reduction='none')
    large_loss = loss_each[loss_each>threshold]
    large_loss_index = torch.nonzero(loss_each>threshold, as_tuple=True)

    return large_loss_index, large_loss


class EarlyStopping:
    """src: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py"""
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, criteria='acc', patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.criteria = criteria
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, score, model):

        if self.criteria == 'acc':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(score, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0

        elif self.criteria == 'loss':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(score, model)
            elif score > self.best_score - self.delta:
                self.counter += 1
                # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0

        else:
            print('criteria is invalid.')

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation acc increased ({self.val_acc_min:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.vval_acc_min = val_acc



def model_train(args, model, adj, feat, labels, idx_train, idx_val, idx_train_p=None, idx_val_p=None, num_simplex=None):
    
    optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.wdecay)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(criteria=args.earlystp_criteria, patience=args.patience, delta=args.delta, verbose=False)

    loss_train_all, loss_val_all, acc_train_all,acc_val_all = [], [], [], []
    dur_all = []
    if args.modelname == 'sccn':
        acc_val_p = torch.zeros(len(num_simplex))
        print('num_simplex: {}'.format(num_simplex))
    

    for ep in range(args.epochs):
        t_ep_start = time.time()
        model.train()
        optimizer.zero_grad()     
        output = model(feat,adj)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
            
        loss_train.backward()
        optimizer.step()

        model.eval()
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if args.earlystp and (ep+1)>args.earlystp_ep_since:
            # due to large val loss caused by special samples, we use acc as the metrics for early stopping
            if args.earlystp_criteria == 'acc':
                early_stopping(acc_val, model) ###
            elif args.earlystp_criteria == 'loss':
                early_stopping(loss_val, model) ###
            else:
                print('invalid criteria.')

            if early_stopping.early_stop:
                print("Early stopping")
                break

        if (ep+1)%200==0:

            print('Epoch: {:04d}'.format(ep+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()))

        loss_train_all.append(loss_train.item())
        loss_val_all.append(loss_val.item())
        acc_train_all.append(acc_train.item())
        acc_val_all.append(acc_val.item())
    
    # load the last checkpoint with the best model
    if args.earlystp:
        if early_stopping.early_stop:
            model.load_state_dict(torch.load('checkpoint.pt'))
            output = model(feat,adj)
    

    return loss_train_all, loss_val_all,acc_train_all,acc_val_all, model, output
    