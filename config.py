import argparse

parser = argparse.ArgumentParser(description='Process hyper-parameters')

parser.add_argument('--dataname', type=str,  default='cora', help='data name')

parser.add_argument('--genmasks', action='store_true', help='True: generate new masks')
parser.add_argument('--pseudospx', action='store_true', help='True: include pseudo simplex in the chains')
parser.add_argument('--earlystp', action='store_true', help='early stopping')

parser.add_argument('--earlystp_criteria',       type=str, default='loss', help='criteria for early stopping (acc or loss)')
# parser.add_argument('--patience',       type=int, default=40, help='patience for early stopping')
parser.add_argument('--patience',           type=int, default=200, help='patience for early stopping')
parser.add_argument('--delta',              type=int, default=0.0001, help='delta for early stopping')
parser.add_argument('--earlystp_ep_since',  type=int, default=800, help='enable early stopping since x epoch')

# parser.add_argument('--modelname', type=str,  default='sccn', help='model name')
parser.add_argument('--modelname', type=str,  default='scn', help='model name')

parser.add_argument('--lr',       type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout',  type=float, default=0.0,   help='dropout ratio')
parser.add_argument('--epochs', type=int,   default=5000, help='epochs num')
parser.add_argument('--hiddim', type=int,   default=16, help='GCN hidden dimension')
parser.add_argument('--wdecay', type=float, default=0e-4, help='Weight decay (L2 loss on parameters).')