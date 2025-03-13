# model.py
import torch,math,time
from torch.nn import Module, Parameter
import torch.nn as nn
from Util import *
import math
import torch.nn.functional as F

class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.unsqueeze(-1)))
        return output

class LocalAggregator(nn.Module):
    def __init__(self, dim, time_encoder, time_dims, alpha, layers, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.time_dims = time_dims
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_4 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.interval_weight_0 = nn.Parameter(torch.randn(self.time_dims, 1))  # for e_0
        self.interval_weight_1 = nn.Parameter(torch.randn(self.time_dims, 1))  # for e_1
        self.interval_weight_2 = nn.Parameter(torch.randn(self.time_dims, 1))  # for e_2
        self.interval_weight_3 = nn.Parameter(torch.randn(self.time_dims, 1))  # for e_3
        self.interval_weight_4 = nn.Parameter(torch.randn(self.time_dims, 1))  # for e_4



        self.leakyrelu = nn.LeakyReLU(alpha)

        self.time_encoder = time_encoder
        



    def forward(self, hidden, adj, A_interval, interval_unique, mask_item):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)
        
        encode_A_interval = self.time_encoder(A_interval).view(batch_size, N, N, self.time_dims)

        interval_weighted_0 = torch.matmul(encode_A_interval , self.interval_weight_0)
        interval_weighted_1 = torch.matmul(encode_A_interval , self.interval_weight_1)
        interval_weighted_2 = torch.matmul(encode_A_interval , self.interval_weight_2)
        interval_weighted_3 = torch.matmul(encode_A_interval , self.interval_weight_3)
        interval_weighted_4 = torch.matmul(encode_A_interval , self.interval_weight_4)

        e_0 = torch.matmul(a_input, self.a_0) + interval_weighted_0
        e_1 = torch.matmul(a_input, self.a_1) + interval_weighted_1
        e_2 = torch.matmul(a_input, self.a_2) + interval_weighted_2
        e_3 = torch.matmul(a_input, self.a_3) + interval_weighted_3
        e_4 = torch.matmul(a_input, self.a_4) + interval_weighted_4

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)
        e_4 = self.leakyrelu(e_4).squeeze(-1).view(batch_size, N, N)
        
        # print(adj.eq(1).shape, e_3.shape, e_0.shape)
        
        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.where(adj.eq(5), e_4, alpha)
        alpha = torch.softmax(alpha, dim=-1)
        return torch.matmul(alpha, h)


class SessionHCov(Module):
    def __init__(self, layers, time_encoder, batch_size, emb_size):
        super(SessionHCov, self).__init__()
        self.time_encoder = time_encoder
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers
        self.neighbor_transfer = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.attn_layer = nn.Linear(self.emb_size, self.emb_size)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.a_n = nn.ParameterList([nn.Parameter(torch.randn(self.emb_size, 1)) for i in range(13)]) # 13 types 
        self.t_n = nn.ParameterList([nn.Parameter(torch.randn(self.emb_size, 1)) for i in range(13)]) # 13 types 
        self.jacard_transform = nn.Linear(1, self.emb_size, bias=False)
        self.sessionT_transform = nn.Linear(1, self.emb_size, bias=False)

    def HyperGAT(self, h, adj, matrix, session_stamp):
        N = h.shape[0] # h: [batch_size, emb_size]
        a_input_1 = h.repeat(1, self.batch_size).view(N * N, self.emb_size)  # Q, shape: [batch_size, batch_size, self.emb_size]
        a_input_2 = h.repeat(self.batch_size, 1)  # K, shape: [batch_size, batch_size, self.emb_size]
        a_input_jac = self.jacard_transform(matrix.unsqueeze(-1))
        # a_input_temporal = self.sessionT_transform(session_stamp.unsqueeze(-1))
        a_input_temporal = session_stamp
        
        # jacard similarity & time difference
        # Element-wise multiplication between session pairs
        a_input = (a_input_1 * a_input_2).view(N, N, self.emb_size)
        e_n = []
        for a_i, t_i in zip(self.a_n, self.t_n):
            t = torch.matmul(a_input_temporal, t_i)
            e = torch.matmul(a_input, a_i)  # shape: [batch_size, batch_size, 1]
            e_n.append(self.LeakyReLU(e + t).squeeze(-1))   # shape: [batch_size, batch_size]

        # Mask to handle padding values or invalid relations
        mask = -9e15 * torch.ones_like(e_n[0])
        
        # Combine adj and e_n based on the adj overlap
        alpha = torch.where(adj.eq(1), e_n[0], mask)
        for i in range(2, len(e_n) + 1):
            alpha = torch.where(adj.eq(i), e_n[i-1], alpha)
        
        alpha = torch.softmax(alpha, dim=-1)
        output = torch.matmul(alpha, h)  # shape: [batch_size, self.emb_size]
        return output

    def forward(self, item_embedding, overlap_type, matrix, session_stamp, session_len):
        item_emb_lgcn = torch.sum(item_embedding.detach(), 1) / session_len  # [batch, emb]
        item_emb_lgcn = self.HyperGAT(item_emb_lgcn, overlap_type, matrix, session_stamp)
        return item_emb_lgcn




class EnHSG(Module):
    def __init__(self, n_node, n_sess, args):
        super(EnHSG, self).__init__()

        self.w_k = 10
        self.n_node = n_node
        self.n_sess = n_sess
        self.lr = args.lr
        self.GRU_inp_len = args.seq_len
        self.hidden_size = args.embSize
        self.time_dims = args.time_dims

        self.layer_num = args.layer
        self.batch_size = args.batchSize
        self.beta = args.beta
        self.emb_size = args.embSize
        self.intent_num = args.intent_num
        self.embedding = trans_to_cuda(nn.Embedding(self.n_node+1, self.emb_size, padding_idx=0))
        # self.pos_embedding = nn.Embedding(200, self.embedding_dim)
        self.args = args

        self.linear_one = nn.Linear(self.time_dims + self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(2 * self.time_dims + self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        
        self.w_1 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        
        self.time_encoder = TimeEncode(self.time_dims)
        self.SHCov = SessionHCov(self.layer_num, self.time_encoder, self.batch_size, self.emb_size)
        self.Type_agg = LocalAggregator(self.hidden_size, self.time_encoder, self.time_dims, 0.2, self.layer_num)
        self.multi_weight =  nn.Parameter(torch.randn(1, self.intent_num, 1 ))
        
        self.similar_linear =  nn.Linear(self.hidden_size + self.time_dims, 1, bias=False)
        self.inter_transfer =  nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.intra_transfer = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.gate_linear = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.final_linear = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=args.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        
        
        self.init_parameters()
        self.dropout = nn.Dropout(args.dropout)

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        self.embedding.weight.data[0].zero_() 

    def generate_session(self, hidden, session_len, interval_full, mask): # attention mechanism
        flipped_interval_full = torch.flip(interval_full, dims=[1]) # flip in the second dimension
        cumsum_interval_full = torch.cumsum(flipped_interval_full, dim=1) # cumulate
        reversed_interval = torch.flip(cumsum_interval_full, dims=[1]) # flip again, reversed interval
        reversed_interval = self.time_encoder(reversed_interval)
        interval_emb = self.time_encoder(interval_full) # [batch, seq]
        
        
        last_pool = list()
        last_interval = list()
        multi_session = list()

        
        for intent in range(1, self.intent_num+1):

            if intent != 1:
                hidden_transposed = hidden.transpose(1, 2)
                hidden_padded = F.pad(hidden_transposed, (intent-1, 0), "constant", 0)
                # Mean pooling
                mean_pooled_hidden = F.avg_pool1d(hidden_padded, kernel_size=intent, stride=1).transpose(1, 2).squeeze(1)
                last_pool.append((mean_pooled_hidden)[torch.arange(mask.shape[0]), torch.sum(mask, 1) - 1])
            else:
                last_pool.append(hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1] )# last intem batch_size x latent_size
            
            start_idx = torch.clamp(torch.sum(mask, 1) - intent, min=0)
            end_idx = torch.sum(mask, 1)
            batch_size, seq_len = interval_full.shape
            indices = torch.arange(seq_len, device=interval_full.device)
            expanded_start_idx = start_idx.unsqueeze(1).expand(batch_size, seq_len)
            expanded_end_idx = end_idx.unsqueeze(1).expand(batch_size, seq_len)
            mask_range = (indices > expanded_start_idx) & (indices <= expanded_end_idx)
            masked_interval = interval_full * mask_range.float()
            last_interval.append(torch.sum(masked_interval, dim=-1))

        for i, (ht, last_int) in enumerate(zip(last_pool, last_interval)):
            last_int_encoded = self.time_encoder(last_int) # [batch, hidden]
            q1 = self.linear_one(torch.cat([last_int_encoded.unsqueeze(1), ht.unsqueeze(1)], -1)) # ht: [batch, hidden]
            q2 = self.linear_two(torch.cat([reversed_interval, interval_emb, hidden], -1))  # batch_size x seq_length x latent_size
            alpha = self.linear_three(torch.sigmoid(q1 + q2))
            gated_output = alpha * hidden * mask.view(mask.shape[0], -1, 1).float()  # b, s, d
            multi_session.append(torch.sum(gated_output, 1))  # [intents, batch, hidden_dim]

        TD_session = torch.sum(torch.softmax(self.multi_weight, dim=1) * torch.stack(multi_session, dim=1), dim=1) 
        return TD_session




    def SSL(self, sess_emb_hgnn, sess_emb_lgcn, tau=0.07):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), dim=1) / tau

        sess_emb_hgnn = F.normalize(sess_emb_hgnn, p=2, dim=1)  # L2 Normalization
        sess_emb_lgcn = F.normalize(sess_emb_lgcn, p=2, dim=1)

        # Compute similarity scores
        pos = score(sess_emb_hgnn, sess_emb_lgcn)  # Positive pairs
        neg = torch.matmul(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn).T) / tau  # Negative pairs
        # Concatenate pos into neg for proper softmax
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)  # Shape: (batch_size, 1 + num_neg_samples)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=sess_emb_hgnn.device)  # Positive pair index is 0
        # Compute InfoNCE loss using CrossEntropy
        SCL_loss = F.cross_entropy(logits, labels)
        return SCL_loss


    def forward(self, full_items, unique_items, alias_inputs, interval_full, session_len, overlap_type,matrix, mask, A_item, A_interval, interval_unique, session_stamp):
        emb = self.embedding(unique_items)
        emb = self.dropout(F.normalize(emb, dim=-1))
        full_emb = self.embedding(full_items)
        time_diff = session_stamp.unsqueeze(0) - session_stamp.unsqueeze(1)  
        time_diff = self.time_encoder(time_diff)
        unique_item_emb_TD = self.Type_agg(emb, A_item, A_interval, interval_unique, mask)
        item_emb_TD = torch.gather(unique_item_emb_TD, dim=1, index=alias_inputs.unsqueeze(-1).expand(-1, -1, unique_item_emb_TD.size(-1)))
        TD_session = self.generate_session(item_emb_TD, session_len, interval_full, mask)
        session_emb_lgcn = self.SHCov(full_emb, overlap_type,matrix, time_diff, session_len)
        b = self.embedding.weight[1:]
        if self.args.dataset!="Yoochoose64":
            b = F.normalize(b, dim=-1)
        session_loss = self.SSL(session_emb_lgcn, TD_session)
        score = torch.matmul(TD_session, b.transpose(1, 0))
        return session_loss * self.beta, score

