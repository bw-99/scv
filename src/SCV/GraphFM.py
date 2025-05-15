import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter as Param
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
import torch
import torch.nn as nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel

def normalize(inputs, eps: float = 1e-8):
    mean = inputs.mean(dim=-1, keepdim=True)
    var = inputs.var(dim=-1, unbiased=False, keepdim=True)
    return (inputs - mean) / torch.sqrt(var + eps)
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter as Param
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
import torch
import torch.nn as nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel

def normalize(inputs, eps: float = 1e-8):
    mean = inputs.mean(dim=-1, keepdim=True)
    var = inputs.var(dim=-1, unbiased=False, keepdim=True)
    return (inputs - mean) / torch.sqrt(var + eps)

class GATAttention(nn.Module):
    def __init__(self, field_size, in_units, out_units, num_heads=1, k=1, has_residual=True, dropout_keep_prob=0.1):
        super(GATAttention, self).__init__()
        self.in_units = in_units
        self.out_units = out_units
        self.num_heads = num_heads
        self.k = k
        self.has_residual = has_residual
        self.att_dense = nn.Linear(in_units, num_heads)
        self.w_dense = nn.Linear(in_units, out_units, bias=False)
        if has_residual:
            self.res_dense = nn.Linear(in_units, out_units)
        # gating scalar net for top-k selection
        self.gsl_1 = nn.Linear(in_units, 16)
        self.gsl_2 = nn.Linear(16, 1)
        self.dropout_keep_prob = dropout_keep_prob

    def forward(self, h):
        B, M, d = h.shape
        # Linear projections per field
        A = F.relu(self.att_dense(h))             # [B, M, H]
        H_lin = self.w_dense(h)                   # [B, M, out_units]
        # Top-k gating per sample
        S_hidden = F.relu(self.gsl_1(h))          # [B, M, 16]
        S = torch.sigmoid(self.gsl_2(S_hidden)).squeeze(-1)  # [B, M]
        # clamp k to number of fields to avoid out-of-range error
        k_eff = min(self.k, S.shape[1])
        topk_vals, _ = torch.topk(S, k_eff, dim=1)  # [B, k_eff]
        kth = topk_vals[:, -1].unsqueeze(1)         # [B, 1]
        mask = (S >= kth).float()                   # [B, M]
        S_mask = mask.unsqueeze(2)                  # [B, M, 1]
        # Apply mask to attention scores
        A_mod = A * S_mask                          # [B, M, H]
        # Split heads
        A_split = torch.split(A_mod, 1, dim=2)      # tuple of H elements [B, M, 1]
        head_dim = self.out_units // self.num_heads
        H_split = torch.split(H_lin, head_dim, dim=2)  # tuple of H elements [B, M, head_dim]
        A_ = torch.cat(A_split, dim=0)              # [H*B, M, 1]
        H_ = torch.cat(H_split, dim=0)              # [H*B, M, head_dim]
        # Compute attention weights
        weights = F.softmax(A_, dim=1)              # [H*B, M, 1]
        weights = F.dropout(weights, p=1-self.dropout_keep_prob, training=self.training)
        # Weighted sum
        out = (weights * H_).sum(dim=1, keepdim=True)  # [H*B, 1, head_dim]
        # Restore shape
        out_split = torch.split(out, B, dim=0)       # tuple H elements [B,1,head_dim]
        out = torch.cat(out_split, dim=2)            # [B,1,out_units]
        # Residual connection
        if self.has_residual:
            Q_res = F.relu(self.res_dense(h.mean(dim=1, keepdim=True)))
            out = out + Q_res
        out = F.relu(out)
        return normalize(out), mask.unsqueeze(1)

# class GraphFM(BaseModel):
#     def __init__(self, feature_map, model_id="GraphFM", gpu=-1,
#                  learning_rate=1e-3, blocks=2, heads=1, ks=None, embedding_regularizer=None,
#                  net_regularizer=None, embedding_dim=10, **kwargs):
#         super(GraphFM, self).__init__(feature_map,
#                                        model_id=model_id,
#                                        gpu=gpu,
#                                        embedding_regularizer=embedding_regularizer,
#                                        net_regularizer=net_regularizer,
#                                        **kwargs)
#         self.blocks = blocks
#         self.heads = heads
#         self.ks = ks or [1] * blocks
#         self.embedding_dim = embedding_dim
#         self.field_size = feature_map.get_num_fields()
#         self.embedding = FeatureEmbedding(feature_map, embedding_dim)
#         # Define one GAT block per layer
#         self.gat_blocks = nn.ModuleList([
#             GATAttention(self.field_size, embedding_dim, num_heads=heads, k=self.ks[i])
#             for i in range(self.blocks)
#         ])
#         total_dim = embedding_dim * self.blocks
#         self.fc_out = nn.Linear(total_dim, 1)

#         self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
#         self.reset_parameters()
#         self.model_to_device()
    
#     def forward(self, inputs):
#         X = self.get_inputs(inputs)
#         # feature embedding: [B, M, d]
#         h = self.embedding(X, flatten_emb=False)
#         # apply GAT blocks
#         out_list = []
#         for i, gat in enumerate(self.gat_blocks):
#             h, vis = gat(h)
#             out_list.append(h)  # [B, 1, d]
#         # concatenate along embedding dim
#         H_cat = torch.cat([o.squeeze(1) for o in out_list], dim=-1)  # [B, blocks*d]
#         logits = self.fc_out(H_cat)
#         y_pred = self.output_activation(logits)
#         return {"y_pred": y_pred, "vis": vis}
# # --- End GraphFM and related functions ---



class GraphFM(BaseModel):
    def __init__(self, feature_map, model_id="GraphFM", gpu=-1,
                 learning_rate=1e-3, heads=1, ks=None, block_shape=None,
                 embedding_regularizer=None, net_regularizer=None, embedding_dim=10,
                 dropout_keep_prob=0.1, **kwargs):
        super(GraphFM, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.blocks = len(block_shape)
        self.heads = heads
        # block_shape: list of output dims for each GAT block
        self.block_shape = block_shape 
        # or [embedding_dim] * blocks
        # ks: top-k for each block (list)
        self.ks = ks if isinstance(ks, list) else [ks] * self.blocks
        self.embedding_dim = embedding_dim
        self.field_size = feature_map.get_num_fields()
        self.embedding = FeatureEmbedding(feature_map, embedding_dim)
        # Define one GAT block per layer
        # in_dims: embedding_dim for block0, then each block_shape[i-1]
        in_dims = [self.embedding_dim] + self.block_shape[:-1]
        self.gat_blocks = nn.ModuleList([
            GATAttention(
                self.field_size,
                in_units=in_dims[i],
                out_units=self.block_shape[i],
                num_heads=heads,
                k=self.ks[i],
                dropout_keep_prob=dropout_keep_prob
            )
            for i in range(self.blocks)
        ])
        total_dim = sum(self.block_shape)
        self.fc_out = nn.Linear(total_dim, 1)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
    
    def forward(self, inputs):
        X = self.get_inputs(inputs)
        # feature embedding: [B, M, d]
        h = self.embedding(X, flatten_emb=False)
        # apply GAT blocks
        out_list = []
        for i, gat in enumerate(self.gat_blocks):
            h, vis = gat(h)
            out_list.append(h)  # [B, 1, d]
        # concatenate along embedding dim
        H_cat = torch.cat([o.squeeze(1) for o in out_list], dim=-1)  # [B, sum(block_shape)]
        logits = self.fc_out(H_cat)
        y_pred = self.output_activation(logits)
        return {"y_pred": y_pred, "vis": vis}
# --- End GraphFM and related functions ---
