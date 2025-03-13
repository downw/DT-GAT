# Util.py
import torch, random
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, find
from sparse import COO
import torch.nn.functional as F
import pickle

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

def get_number_of_items(item_list):
    unique_items = set(item_list)
    return len(unique_items)

def transfer_df2list(df):
    two_dimensional_list = df.groupby('session_id')['item_id'].apply(list).tolist()
    return two_dimensional_list


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def data_masks(all_train_seq, seq_inputs, all_interval_seqs, item_tail, interval_tail):
    allseq_lens = [len(allseq) for allseq in all_train_seq]
    alllen_max = max(allseq_lens)
    seq_lens = [len(seq) for seq in seq_inputs]

    all_train_seq =  [allseq + item_tail * (alllen_max - le) for allseq, le in zip(all_train_seq, allseq_lens)]
    interaction_seqs = [seq + item_tail * (alllen_max - le) for seq, le in zip(seq_inputs, seq_lens)]
    interval_seqs = [seq + interval_tail * (alllen_max - le) for seq, le in zip(all_interval_seqs, seq_lens)]
    seq_masks = [[1] * le + [0] * (alllen_max - le) for le in seq_lens]
    return all_train_seq, interaction_seqs, interval_seqs, seq_masks


    



class Data():
    def __init__(self, data, name, all_train_seq, args, shuffle=False, n_node=None):
        print("------------start loading data-------------------")
        self.seq_inputs = data[0]
        self.interval_inputs = data[1]
        all_train_seq, seq_inputs, interval_inputs, mask = data_masks(all_train_seq, self.seq_inputs, self.interval_inputs, [0], [0])
        self.all_train_seq = np.asarray(all_train_seq)
        self.seq_inputs = np.asarray(seq_inputs)
        self.session_id = np.arange(self.all_train_seq.shape[0])
        self.interval_inputs = np.asarray(interval_inputs)
        self.mask = np.asarray(mask)
        self.args = args
        self.targets = np.asarray(data[2])
        self.session_stamp = np.asarray(data[3])
        self.length = len(self.seq_inputs)
        self.shuffle = shuffle
    
    def load_from_file(self, filename):
        """
        Load data from a file using pickle.
        """
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        print(f"Data loaded from {filename}")
        return data

    def calculate_jaccard_similarity(self, matrix_A, matrix_B):
        """
        Calculate the Jaccard similarity between two binary sparse matrices.
        matrix_A: csr_matrix of shape (n_samples_A, n_features)
        matrix_B: csr_matrix of shape (n_samples_B, n_features)
        
        Returns a similarity matrix of shape (n_samples_A, n_samples_B)
        """
        # Calculate the intersection: shared elements between A and B
        intersection = matrix_A.dot(matrix_B.T).tocsc()  # Intersection count (dot product works for sparse matrices)

        # Row sums represent the number of non-zero elements in each sample (for both A and B)
        row_sums_A = np.array(matrix_A.sum(axis=1)).flatten()  # Number of elements in each row of A
        row_sums_B = np.array(matrix_B.sum(axis=1)).flatten()  # Number of elements in each row of B

        # Calculate union: A + B - A ∩ B
        union = row_sums_A[:, None] + row_sums_B[None, :] - intersection.toarray()  # Convert to dense array (locally)

        # Prevent division by zero
        union[union == 0] = 1

        # Jaccard similarity: intersection / union
        jaccard_similarity = intersection.toarray() / union
        
        return jaccard_similarity



    def nonzero_indices(self, crs_matrix):
        nonzero_matrix = find(crs_matrix)
        return nonzero_matrix[0], nonzero_matrix[1], nonzero_matrix[2]
    
    def get_overlap(self, sessions, session_len, session_stamp):
        overlap_type = np.zeros((len(sessions), len(sessions)))
        matrix = np.zeros((len(sessions), len(sessions)))

        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)  # Remove padding items (e.g., 0)
            stamp_a = session_stamp[i]
            
            for j in range(i + 1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)  # Remove padding items (e.g., 0)
                stamp_b = session_stamp[j]
                
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b  # Union of the two sets
                
                # Calculate Jaccard similarity (intersection over union)
                matrix[i][j] = float(len(overlap)) / float(len(ab_set)) if len(ab_set) > 0 else 0.0
                matrix[j][i] = matrix[i][j]

                # Determine overlap relationship
                if len(overlap) == 0:
                    # No overlap
                    overlap_relation_i_to_j = 0
                    overlap_relation_j_to_i = 0
                elif seq_a.issubset(seq_b):
                    overlap_relation_i_to_j = 1  # i is a subset of j
                    overlap_relation_j_to_i = 2  # j is a superset of i
                elif seq_b.issubset(seq_a):
                    overlap_relation_i_to_j = 2  # i is a superset of j
                    overlap_relation_j_to_i = 1  # j is a subset of i
                else:
                    overlap_relation_i_to_j = 3  # Overlap but not a subset
                    overlap_relation_j_to_i = 3

                # Determine time relationship
                if stamp_a == stamp_b:
                    time_relation_i_to_j = 0  # Same timestamp
                    time_relation_j_to_i = 0  # Same timestamp in reverse direction
                elif stamp_a < stamp_b:
                    time_relation_i_to_j = 1  # Current session happens earlier
                    time_relation_j_to_i = 2  # Reverse, current session happens later
                else:
                    time_relation_i_to_j = 2  # Current session happens later
                    time_relation_j_to_i = 1  # Reverse, current session happens earlier

                # Combine overlap and time relationship (9 possible cases)
                overlap_type[i][j] = overlap_relation_i_to_j * 3 + time_relation_i_to_j
                overlap_type[j][i] = overlap_relation_j_to_i * 3 + time_relation_j_to_i

        matrix = matrix + np.diag([1.0] * len(sessions))  # Add self-loop (diagonal of 1.0)
        
        
        overlap_type = overlap_type + np.diag([12] * len(sessions))  # Add self-loop type (diagonal of 12), total number of types is 13
        return overlap_type, matrix
    
    
    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.seq_inputs = self.seq_inputs[shuffled_arg]
            self.interval_inputs = self.interval_inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            self.session_stamp = self.session_stamp[shuffled_arg]
            # self.similar_sessions = self.similar_sessions[shuffled_arg]

        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        # similar_seqs, similarity, seq_inputs, interval_inputs, mask, targets, session_stamp = self.all_train_seq[self.similar_sessions[index]], self.jaccard_similarities[index], self.seq_inputs[index], self.interval_inputs[index], self.mask[index], self.targets[index], self.session_stamp[index]
        seq_inputs, interval_inputs, mask, targets, session_stamp = self.seq_inputs[index], self.interval_inputs[index], self.mask[index], self.targets[index], self.session_stamp[index]
        unique_items, alias_inputs, session_len, n_node, A_item, A_interval, interval_unique, full_items, interval_full, last_mode, last_item, last_inetrval = [], [], [], [], [], [], [], [], [], [], [], []
        session_stamp = session_stamp - np.min(session_stamp)
        interval_inputs = interval_inputs / 1e3
        session_stamp = (session_stamp / 86400).astype(int)
        for u_input in seq_inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for seq, intervals in zip(seq_inputs, interval_inputs):
            node = np.unique(seq)
            unique_items.append(node.tolist() + (max_n_node - len(node)) * [0])
            adj = np.zeros((max_n_node, max_n_node)) # adjacency
            i_A = np.zeros((max_n_node, max_n_node)) # inetrval matrix
            unique_interval = np.zeros(max_n_node) # one-dimension unique_interval
            for i, interval in zip(np.arange(len(seq[seq!=0]) - 1), intervals):
                u = np.where(node == seq[i])[0][0]
                adj[u][u] = 1
                if seq[i + 1] == 0:
                    break
                v = np.where(node == seq[i + 1])[0][0]
                i_A[u][v] = interval
                if u == v or adj[u][v] == 4:
                    continue
                adj[v][v] = 1
                if adj[v][u] == 2:
                    adj[u][v] = 4
                    adj[v][u] = 4
                else:
                    adj[u][v] = 2
                    adj[v][u] = 3
                # Handle two-hop neighbors
                if i + 2 < len(seq[seq != 0]):  
                    w = np.where(node == seq[i + 2])[0][0]  # Find the 2-hop node w
                    if adj[u][w] == 0:  # Check that u->w and w->v are not connected
                        adj[u][w] = 5 
                        i_A[u][w] = intervals[i + 1] 
                
                unique_interval[u] = interval
            A_item.append(adj)
            A_interval.append(i_A)
            session_len.append([len(seq[seq != 0])])
            interval_unique.append(unique_interval)
            alias_inputs.append([np.where(node == i)[0][0] for i in seq])
            full_items.append(seq)
            interval_full.append(intervals)
        return full_items, unique_items, alias_inputs, session_len, interval_full, A_item, A_interval, interval_unique, mask, targets, session_stamp
