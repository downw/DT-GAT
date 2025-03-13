# main.py
import argparse
import numpy as np
import pandas as pd
import pickle
import time, os
from model import *
from train import *
from Util import *


script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/Retailrocket/Yoochoose64')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--seq_len', type=int, default=100, help='sequence length for GRU input')

parser.add_argument('--embSize', type=int, default=100, help='embedding size of item & session')
parser.add_argument('--time_dims', type=int, default=100, help='embedding size of time')

parser.add_argument('--l2', type=float, default=1e-4, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=10, help='the number of steps after which the learning rate decay')
parser.add_argument('--layer', type=float, default=1, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.005, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU for acceleration')
parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
parser.add_argument('--intent_num', type=int, default=4, help='number of intents')

parser.add_argument('--random_seed', type=int, default=0, help='random seed')
parser.add_argument('--dropout', type=float, default=0, help='dropout')

args = parser.parse_args()
print(args)


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.cuda.set_device(0)

def main():
    setup_seed(args.random_seed)
    data_directory = 'processed_data'
   



    train_seqs = pickle.load(open(os.path.join("datasets", args.dataset , 'processed_data/train.txt'), "rb"))
    test_seqs = pickle.load(open(os.path.join("datasets", args.dataset , 'processed_data/test.txt'), "rb"))
    all_train = pickle.load(open(os.path.join("datasets", args.dataset  , 'processed_data/all_train_seq.txt'), "rb"))

    n_sess = len(train_seqs[1])+len(test_seqs[1])
    if args.dataset == 'diginetica':
        n_items = 43098
    elif args.dataset == 'Retailrocket': # avg 6.71
        n_items = 50020
    elif args.dataset == 'Nowplaying':
        n_items = 60418
    elif args.dataset == 'Yoochoose64':
        n_items = 37485
    elif args.dataset == 'Yoochoose' or args.dataset == 'Yoochoose4':
        n_items = 37485
    # train_seqs = process_seqs(transfer_df2list(train_data))
    # test_seqs =  process_seqs(transfer_df2list(test_data))
    print(len(train_seqs[0]))
    print(len(test_seqs[0]))
    assert len(train_seqs) == 4, "the form of train_seqs is incorrect" #([item sequence], [time interval sequence], [target item])
    
    train_data = Data(train_seqs, "train", all_train, args, shuffle=True, n_node=n_items)
    test_data = Data(test_seqs, "test", all_train, args, shuffle=True, n_node=n_items)
    model = trans_to_cuda(EnHSG(n_items, n_sess, args)) 
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]
    bad_counter = 0
    
    for epoch in range(args.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)


        metrics, total_loss, total_ssl_loss = train_test(model, train_data, test_data)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        print(metrics)

        for K in top_K:
            print('train_s:\t%.4f\lostHit@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
    
    
    
if __name__ == '__main__':
    main()
