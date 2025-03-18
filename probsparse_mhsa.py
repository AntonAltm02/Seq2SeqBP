import math
import numpy as np
import torch
import torch.nn as nn


class FullAttention(nn.Module):
    def __init__(self, mask_flag, factor, scale=None, attn_drop=0.1, output_attn=False):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attn_drop)
        self.output_attn = output_attn

    def forward(self, q, k, v, attn_mask):
        B, L, H, D = q.shape
        _, S, _, _ = k.shape
        scale = self.scale or 1. / math.sqrt(D)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = self.dropout(torch.softmax(scale * attn_scores, dim=-1))

        attention = torch.matmul(attn_scores, v)

        return attention


class ProbAttention(nn.Module):
    def __init__(self, mask_flag, factor, scale=None, attn_drop=0.1, output_attn=False):
        super(ProbAttention, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attn_drop)
        self.output_attn = output_attn

    def _prob_qk(self, q, k, sample_k, u_top):
        B, H, L_q, D = q.shape
        _, _, L_k, _ = k.shape

        k_exp = k.unsqueeze(-3).expand(B, H, L_q, L_k, D)
        idx_sample = torch.randint(L_k, (L_q, sample_k))
        k_sample = k_exp[:, :, torch.arange(L_q).unsqueeze(1), idx_sample, :]
        qk_sample = torch.matmul(q.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze(-2)

        # finding top-u query with sparsity measurement
        M = qk_sample.max(-1)[0] - torch.div(qk_sample.sum(-1), L_k)
        M_top = M.topk(u_top, sorted=False)[1]

        # calculate Q_reduce as query_states[:, M_top]
        # use the reduced Q to calculate Q_K
        q_red = q[torch.arange(B)[:, None, None],
                torch.arange(H)[None, :, None],
                M_top, :]  # factor*ln(L_q)  # size: c*log_L_Q x channel

        return q_red, M_top

    def _update_context(self, v, scores, idx):
        B, H, L_v, D = v.shape

        attn_scores = torch.softmax(scores, dim=-1)

        v[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], idx, :] = \
            torch.matmul(attn_scores, v).type_as(v)

        return v

    def forward(self, q, k, v, attn_mask):
        B, L_q, H, D = q.shape
        _, L_k, _, _ = k.shape

        # change shape to (B, H, L, D)
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)
        v = v.transpose(2, 1)

        # calc: u_part = c * ln(l_k)
        u_part = self.factor * np.ceil(np.log(L_k)).astype("int").item()
        # calc: u = c * ln(l_q)
        u = self.factor * np.ceil(np.log(L_q)).astype("int").item()

        u_part = u_part if u_part < L_k else L_k  # ensure that u_part is smaller than l_k
        u = u if u < L_q else L_q  # ensure that u is smaller than l_q

        q_red, M_top = self._prob_qk(q, k, sample_k=u_part, u_top=u)

        sparse_attn_scores = torch.matmul(q_red, k.transpose(-2, -1))
        sparse_attn_scores = sparse_attn_scores / math.sqrt(D)

        attn_probs = self._update_context(v, sparse_attn_scores, M_top)

        return attn_probs.transpose(2, 1)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, num_head):
        super(AttentionLayer, self).__init__()

        dim_k = (d_model // num_head)
        dim_v = (d_model // num_head)
        self.num_head = num_head

        self.attention = attention
        self.q_linear = nn.Linear(d_model, dim_k * num_head)
        self.k_linear = nn.Linear(d_model, dim_k * num_head)
        self.v_linear = nn.Linear(d_model, dim_v * num_head)
        self.out_linear = nn.Linear(dim_v * num_head, d_model)

    def forward(self, query, key, value, attn_mask=None):
        B, L, _ = query.shape
        _, S, _ = key.shape
        H = self.num_head

        queries = self.q_linear(query).view(B, L, H, -1)
        keys = self.k_linear(key).view(B, S, H, -1)
        values = self.v_linear(value).view(B, S, H, -1)

        out = self.attention(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, S, -1)
        return self.out_linear(out)
