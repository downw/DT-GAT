# train.py
import datetime,torch
import numpy as np
from Util import *
import heapq
from numba import jit
from tqdm import tqdm

@jit
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    # k_largest_scores = [item[0] for item in n_candidates]
    return ids#, k_largest_scores

def train_batch(model, i, data):
    full_items, unique_items, alias_inputs, session_len, interval_full, A_item, A_interval, interval_unique, mask, tar, session_stamp = data.get_slice(i)
    full_items = np.array(full_items)
    session_len = np.array(session_len)
    session_stamp = np.array(session_stamp)
    overlap_type, matrix = data.get_overlap(full_items, session_len, session_stamp)
    matrix = trans_to_cuda(torch.Tensor(matrix))
    overlap_type = trans_to_cuda(torch.Tensor(overlap_type))

    session_len = trans_to_cuda(torch.Tensor(session_len).float())
    interval_full = trans_to_cuda(torch.Tensor(interval_full).float())
    A_item = trans_to_cuda(torch.tensor(np.array(A_item)).float())
    A_interval = trans_to_cuda(torch.tensor(np.array(A_interval)).float())
    interval_unique = trans_to_cuda(torch.Tensor(np.array(interval_unique)).float())
    
    # print("Before CUDA:", torch.Tensor(sessions).shape)
    full_items = trans_to_cuda(torch.Tensor(full_items).long()) # [batch, max_seq]
    unique_items = trans_to_cuda(torch.Tensor(np.array(unique_items)).long())
    alias_inputs = trans_to_cuda(torch.Tensor(np.array(alias_inputs)).long())
    session_stamp = trans_to_cuda(torch.tensor(session_stamp).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    
    ssl_loss, scores = model(full_items, unique_items, alias_inputs, interval_full, session_len, overlap_type,matrix,
                             mask, A_item, A_interval, interval_unique, session_stamp)
    return ssl_loss, tar, scores 

def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    total_ssl_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for index, i in tqdm(enumerate(slices),total=len(slices)):
        model.optimizer.zero_grad()
        ssl_loss, targets, scores  = train_batch(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss = loss + ssl_loss
        loss.backward()
        model.optimizer.step()
        total_ssl_loss += ssl_loss
        total_loss += loss
        if index % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (index, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    
    model.eval()

    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        ssl_loss, targets, scores = train_batch(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        # targets = trans_to_cpu(targets).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], targets):
                metrics['hit%d' %K].append(np.isin(target-1, prediction))
                if len(np.where(prediction == target - 1)[0]) == 0:
                    metrics['mrr%d' %K].append(0)
                else:
                    metrics['mrr%d' %K].append(1 / (np.where(prediction == target - 1)[0][0]+1))
    return metrics, total_loss, total_ssl_loss