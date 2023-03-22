import torch
import numpy as np
from utils.data_utils import load_data_sc, gen_masks
from utils.train_utils import accuracy, model_train
from utils.models import GCN
from config import parser


def main():

    args = parser.parse_args()

    # generate train/val/test masks. here we use val as test (no validation, train and test only)
    # for fair comparation, each time the mask is randomly generated and it's not fixed
    if args.genmasks:
        print('generating new masks')
        gen_masks(args.dataname, args.modelname, args.pseudospx)


    if args.modelname == 'scn':
        # simplicial graph convolutional networks (model and train for simplex chains of each dimension)
        print('model: SCN')
        
        adj, feat, labels, chains, idx_train, idx_val, idx_test \
            = load_data_sc(args.dataname, args.modelname, args.pseudospx)

        l_train_all, l_val_all,ac_train_all,ac_val_all = [], [], [], []
        acc_test = [2 for _ in range(len(adj))] # if 2, means training is skipped
        num_test = [0 for _ in range(len(adj))]

        for p in range(len(adj)): # train GCN model for each simplex dimension p
            
            print('Dimension = {:01d}, Number of simplices = {}'.format(p, adj[p].shape[0]))

            if adj[p].shape[0] < 10:
                print('total sample size is less than 10, skip training')
            else:

                model = GCN(nfeat=feat[p].shape[1],
                            nhid = args.hiddim,
                            nout = labels[p].max().item() + 1,
                            dropout = args.dropout)
                            
                ###
                model = model.double()
                feat[p] = feat[p].double()
                adj[p] = adj[p].double()

                if torch.any(idx_train[p]) and torch.any(idx_val[p]):
                    loss_train_all, loss_val_all, acc_train_all, acc_val_all, model, output\
                        = model_train(args, model, adj[p], feat[p], labels[p], idx_train[p], idx_val[p])

                l_train_all.append({p: loss_train_all})
                l_val_all.append({p: loss_val_all})
                ac_train_all.append({p: acc_train_all})
                ac_val_all.append({p: acc_val_all})

                model.eval()
                acc_test[p] = accuracy(output[idx_test[p]], labels[p][idx_test[p]])
                num_test[p] = output[idx_test[p]].shape[0]
                print('acc_test_p[{}]: {:.4f}, num_test_p[{}]: {}'.format(p, acc_test[p].item(), p, num_test[p]))
        
        acc_test_all = sum([acc_test[p] * num_test[p] for p in range(len(adj))]) / sum(num_test)
        print('acc_test_all: {:.4f}'.format(acc_test_all))
        
        print('end training for {}\n'.format(args.modelname))



    elif args.modelname == 'sccn':
        # simplicial complex convolutional networks (model and train for simplex of all dimension)
        print('model: SCCN')

        adj, feat, labels, idx_train, idx_val, idx_test, idx_train_p, idx_val_p, idx_test_p, num_simplex \
            = load_data_sc(args.dataname, args.modelname, args.pseudospx)

        if labels.shape[0] < 100:
            print('total sample size is less than 100, skip training...')
        else:
            model = GCN(nfeat=feat.shape[1],
                            nhid = args.hiddim,
                            nout = labels.max().item() + 1,
                            dropout = args.dropout)
                            
            ###
            model = model.double()
            feat = feat.double()
            adj = adj.double()

            if torch.any(idx_train) and torch.any(idx_val):
                loss_train_all, loss_val_all, acc_train_all, acc_val_all, model, output \
                    = model_train(args, model, adj, feat, labels, idx_train, idx_val, idx_train_p, idx_val_p, num_simplex)

            print('true label', np.histogram(labels[idx_train],[0,1,2,3,4],density=True))
            print('predicted train label', np.histogram(output[idx_train].max(1)[1],[0,1,2,3,4],density=True))


            model.eval()
            acc_test = accuracy(output[idx_test], labels[idx_test])
            acc_test_p = torch.zeros(len(num_simplex))
            for p in range(len(num_simplex)):

                if num_simplex[p]>0:
                    acc_test_p[p] = accuracy(output[idx_test_p[p]], labels[idx_test_p[p]])
                else:
                    acc_test_p[p] = 0
            print('acc_test_p: {}'.format(acc_test_p),
                  'acc_test: {:.4f}'.format(acc_test.item()))

            print('end training for {}\n'.format(args.modelname))



if __name__=='__main__':
    main()