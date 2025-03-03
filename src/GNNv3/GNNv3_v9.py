import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import math
from torch.nn import Parameter as Param
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_regularizer
import numpy as np
import sys
import logging
from tqdm import tqdm
import tempfile
from pathlib import Path
import ray.cloudpickle as pickle
from ray import tune, train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler


class MyGRUCell(nn.Module):
    """
    PyTorch-style GRUCell with manual parameter definitions.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_ih = Param(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Param(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Param(torch.Tensor(3 * hidden_size))
            self.bias_hh = Param(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x: Tensor, hx: Tensor) -> Tensor:
        """
        x: (N, input_size)
        hx: (N, hidden_size)
        """
        if hx is None:
            hx = x.new_zeros(x.size(0), self.hidden_size)

        gates = F.linear(x, self.weight_ih, self.bias_ih) \
              + F.linear(hx, self.weight_hh, self.bias_hh)
        r, z, n = gates.chunk(3, dim=1)

        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(r * n)

        hy = (1 - z) * n + z * hx
        return hy


class CrossNetwork(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 layer_norm=True,
                 batch_norm=True,
                 use_same_adj=False,
                 num_mask=2,
                 pooling_type="mean",
                 pooling_dim=2,
                 gpu=0,
                 nomalize_adj=True,
                 num_tower=2,
                 net_dropout=0.1,
                 num_hops=1):
        super(CrossNetwork, self).__init__()
        self.num_tower = num_tower
        self.nomalize_adj = nomalize_adj
        self.use_same_adj = use_same_adj
        self.num_mask = num_mask
        self.layer_norm_flag = layer_norm
        self.batch_norm_flag = batch_norm
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.num_hops = num_hops

        if self.use_same_adj:
            # shape = (num_tower, num_mask, num_fields, num_fields)
            self.masker = nn.Parameter(torch.zeros(
                (self.num_tower, self.num_mask, num_fields, num_fields)
            ))
        else:
            # shape = (num_tower, num_hops, num_mask, num_fields, num_fields)
            self.masker = nn.Parameter(torch.zeros(
                (self.num_tower, self.num_hops, self.num_mask, num_fields, num_fields)
            ))
        nn.init.xavier_uniform_(self.masker)

        # Weights for each hop
        # shape = (num_hops, embedding_dim, embedding_dim)
        self.weight = Param(torch.Tensor(num_hops, embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # A single GRUCell shared across all hops
        self.rnn = MyGRUCell(embedding_dim, embedding_dim, bias=True)

        if layer_norm:
            self.layer_norm = nn.LayerNorm([self.num_tower, num_fields, embedding_dim])

        if self.batch_norm_flag:
            out_dim = self.num_tower * self.num_fields * self.embedding_dim
            self.batch_norm = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(net_dropout) if net_dropout > 0 else None

        self.pool = GlobalPooling(pooling_type, pooling_dim=pooling_dim)
        
        # Query, Key, Value projection
        self.query_layer = nn.Linear(embedding_dim, embedding_dim)
        self.key_layer = nn.Linear(embedding_dim, embedding_dim)
        self.value_layer = nn.Linear(embedding_dim, embedding_dim)

        # Output projection after multi-head attention
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.num_heads = 8
        self.head_dim = embedding_dim // self.num_heads
        assert embedding_dim % self.num_heads == 0, "embedding_dim must be divisible by num_heads."


    def forward(self, x):
        """
        x: (batch_size, num_fields, embedding_dim)
        Output: (batch_size, num_tower * embedding_dim)
        """
        batch_size, num_fields, embedding_dim = x.size()

        # Expand x for multiple towers => (B, T, N, D)
        x = x.unsqueeze(1).repeat(1, self.num_tower, 1, 1)

        if self.use_same_adj:
            # shape => (num_tower, num_mask, N, N)
            # => reduce mask dimension => (num_tower, N, N)
            adj_matrix = torch.prod(self.masker, dim=1)  # (T, N, N)
            adj_matrix = F.relu(adj_matrix)
            # expand for batch => (B, T, N, N)
            adj_matrix = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1, -1)

            for idx in range(self.num_hops):
                # 1) Linear transform
                m = torch.matmul(x, self.weight[idx])  # still (B, T, N, D)

                # 2) GNN adjacency: (B, T, N, N) @ (B, T, N, D) => (B, T, N, D)
                m = adj_matrix @ m

                # 3) Flatten so MyGRUCell can handle (N, D)
                #    We'll treat (B, T, N) as a single dimension => B*T*N
                B, T, N, D = m.shape
                m_2d = m.view(-1, D)
                x_2d = x.view(-1, D)
                x_2d = self.rnn(m_2d, x_2d)
                x = x_2d.view(B, T, N, D)

        else:
            # shape => (num_tower, num_hops, num_mask, N, N)
            for idx in range(self.num_hops):
                # (T, num_mask, N, N)
                adj_matrix_hop = self.masker[:, idx, ...]
                adj_matrix_hop = torch.prod(adj_matrix_hop, dim=1)  # => (T, N, N)
                adj_matrix_hop = F.relu(adj_matrix_hop)
                # expand => (B, T, N, N)
                adj_matrix_hop = adj_matrix_hop.unsqueeze(0).expand(x.size(0), -1, -1, -1)

                # 1) Linear transform => (B, T, N, D)
                m = torch.matmul(x, self.weight[idx])
                # 2) Adjacency => (B, T, N, N) @ (B, T, N, D) => (B, T, N, D)
                m = adj_matrix_hop @ m

                # 3) Flatten for MyGRUCell
                B, T, N, D = m.shape
                m_2d = m.view(-1, D)      # (B*T*N, D)
                x_2d = x.view(-1, D)      # (B*T*N, D)
                x_2d = self.rnn(m_2d, x_2d)
                x = x_2d.view(B, T, N, D)

        # Apply batch_norm if needed
        if self.batch_norm_flag:
            x_reshaped = x.view(x.size(0), -1)  # => (B, T*N*D)
            x_reshaped = self.batch_norm(x_reshaped)
            x = x_reshaped.view(x.size(0), self.num_tower, self.num_fields, self.embedding_dim)

        # LayerNorm
        if self.layer_norm_flag:
            x = self.layer_norm(x)  # LN over (B, T, N, D)

        # Dropout
        if self.dropout:
            x = self.dropout(x)

        # Now pool across fields dimension => (B, T, D)
        # x shape: (batch_size, num_fields, embedding_dim)
        B, T, N, D = x.size()
        # x = x.view(-1, self.num_fields, self.embedding_dim)
        
        # Linear projections
        Q = self.query_layer(x) # (B, T, N, D)
        K = self.key_layer(x)   # (B, T, N, D)
        V = self.value_layer(x) # (B, T, N, D)

        # Split into multiple heads
        # Resulting shape: (B, T, num_heads, N, head_dim)
        Q = Q.view(B, T, self.num_heads, N, self.head_dim).transpose(2, 3)
        K = K.view(B, T, self.num_heads, N, self.head_dim).transpose(2, 3)
        V = V.view(B, T, self.num_heads, N, self.head_dim).transpose(2, 3)

        # Scaled dot-product attention
        # attn_weights: (B, T, num_heads, N, N)
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn_weights, dim=-1)

        # Apply attention to V
        # out: (B, num_heads, N, head_dim)
        out = torch.matmul(attn, V)

        # Concat heads
        out = out.transpose(2, 3).contiguous().view(B, T, N, D)

        # Final projection and fusion
        out = self.out_proj(out).view(B, T, N, -1) # (B, T, N, D)
        # Flatten before fusion network
        out = out.mean(dim=2)
        return out


class GNNv3_v9(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="GNNv3_v9",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0.1,
                 num_tower=2,
                 num_mask=2,
                 num_hops=1,
                 use_same_adj=False,
                 nomalize_adj=True,
                 layer_norm=False,
                 batch_norm=False,
                 pooling_dim=2,
                 pooling_type="mean",
                 fusion_type="MLP",
                 embedding_regularizer=None,
                 parallel_dnn_hidden_units=[400,400,400],
                 net_regularizer=None,
                 **kwargs):
        super(GNNv3_v9, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.num_tower = num_tower
        self.pooling_dim = pooling_dim
        self.num_tower = num_tower
        self.nomalize_adj = nomalize_adj
        self.num_mask = num_mask
        self.use_same_adj = use_same_adj
        self.gpu = gpu
        self.num_hops = num_hops

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()

        print("num fields", feature_map.get_num_fields())
        self.num_fields = feature_map.get_num_fields()
        self.input_dim = input_dim
        print("GNNv3_v9 input_dim", input_dim)

        self.gnn_tower = CrossNetwork(
            num_fields=self.num_fields,
            embedding_dim=embedding_dim,
            net_dropout=net_dropout,
            num_tower=num_tower,
            num_mask=self.num_mask,
            layer_norm=layer_norm,
            pooling_type=pooling_type,
            use_same_adj=self.use_same_adj,
            pooling_dim=pooling_dim,
            batch_norm=batch_norm,
            gpu=self.gpu,
            nomalize_adj=self.nomalize_adj,
            num_hops=num_hops
        )

        final_dim = embedding_dim

        self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                      output_dim=final_dim,
                                      hidden_units=parallel_dnn_hidden_units,
                                      hidden_activations="ReLU",
                                      output_activation=None,
                                      dropout_rates=net_dropout,
                                      batch_norm=batch_norm)

        # Combine GNN output + parallel DNN => (num_tower + 1) heads
        concat_dim = (self.num_tower + 1) * final_dim

        # Dynamically construct a fusion module
        self.scorer = globals()[f"Fusion{fusion_type}"](
            num_fields=self.num_tower+1,
            embedding_dim=embedding_dim,
            net_dropout=net_dropout,
            num_tower=self.num_tower,
            num_mask=self.num_mask,
            layer_norm=layer_norm,
            pooling_type=pooling_type,
            use_same_adj=self.use_same_adj,
            pooling_dim=pooling_dim,
            batch_norm=batch_norm,
            gpu=self.gpu,
            nomalize_adj=self.nomalize_adj,
            num_hops=num_hops,
        )

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        print("without emb dim")
        self.count_parameters(count_embedding=False)

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=False)
        # feature_emb => (batch_size, num_fields, embedding_dim)
        flattened_emb = feature_emb.view(feature_emb.shape[0], -1)

        # GNN part
        gnn_emb = self.gnn_tower(feature_emb)            # (B, num_tower * embedding_dim)

        # DNN part
        dnn_emb = self.parallel_dnn(flattened_emb).unsqueeze(dim=1)       # (B, embedding_dim)

        # Merge => (B, (num_tower+1)*embedding_dim)
        final_emb = torch.cat([gnn_emb, dnn_emb], dim=1)
        y_pred = self.scorer(final_emb)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


class GlobalPooling(nn.Module):
    """
    Dummy global pooling example for demonstration.
    You may have your own pooling logic.
    """
    def __init__(self, pooling_type="mean", pooling_dim=2):
        super().__init__()
        self.pooling_type = pooling_type
        self.pooling_dim = pooling_dim

    def forward(self, x):
        # You can customize this. Currently unused in CrossNetwork above,
        # because x.mean(dim=2) is used directly.
        return x


class FusionMLP(nn.Module):
    def __init__(self, embedding_dim, num_fields, **kwargs):
        super(FusionMLP, self).__init__()
        concat_dim = (num_fields) * embedding_dim
        self.fusion_network = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, 1),
        )

    def forward(self, x):
        return self.fusion_network(x)

class FusionMaxPooling(nn.Module):
    def __init__(self, embedding_dim, num_fields, **kwargs):
        super(FusionMaxPooling, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        concat_dim = embedding_dim
        self.scorer = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, 1),
        )

    def forward(self, x):
        x = x.view(-1, self.num_fields, self.embedding_dim)
        return self.scorer(torch.max(x, dim=1)[0])

class FusionMeanPooling(nn.Module):
    def __init__(self, embedding_dim, num_fields, **kwargs):
        super(FusionMeanPooling, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        concat_dim = embedding_dim
        self.scorer = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, 1),
        )

    def forward(self, x):
        x = x.view(-1, self.num_fields, self.embedding_dim)
        return self.scorer(torch.mean(x, dim=1))

class FusionATTN(nn.Module):
    def __init__(self, embedding_dim, num_fields, num_heads=8, **kwargs):
        super(FusionATTN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads."

        concat_dim = num_fields * embedding_dim

        # Query, Key, Value projection
        self.query_layer = nn.Linear(embedding_dim, embedding_dim)
        self.key_layer = nn.Linear(embedding_dim, embedding_dim)
        self.value_layer = nn.Linear(embedding_dim, embedding_dim)

        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Final fusion
        self.fusion_network = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, 1),
        )

    def forward(self, x):
        # x: (B, num_fields, embedding_dim)
        B = x.shape[0]
        # x = x.view(B, self.num_fields, self.embedding_dim)

        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        # Split heads => (B, heads, N, head_dim)
        Q = Q.view(B, self.num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, self.num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, self.num_fields, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn, V)  # (B, heads, N, head_dim)

        # Concat heads => (B, N, D)
        out = out.transpose(1, 2).contiguous().view(B, self.num_fields, self.embedding_dim)
        out = self.out_proj(out)

        # Flatten => (B, N*D)
        out = out.view(B, -1)
        out = self.fusion_network(out)
        return out


class FusionGAS(nn.Module):
    def __init__(self, 
                 num_fields,
                 embedding_dim,
                 layer_norm=True,
                 batch_norm=True,
                 use_same_adj=False,
                 num_mask=2,
                 pooling_type="mean",
                 pooling_dim=2,
                 gpu=0,
                 nomalize_adj=True,
                 num_tower=2,
                 net_dropout=0.1,
                 num_hops=1, 
                 **kawrgs):
        super(FusionGAS, self).__init__()
        self.fusion_netowrk = CrossNetwork(
            num_fields,
            embedding_dim,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            use_same_adj=use_same_adj,
            num_mask=num_mask,
            pooling_type=pooling_type,
            pooling_dim=pooling_dim,
            gpu=gpu,
            nomalize_adj=nomalize_adj,
            num_tower=num_tower,
            net_dropout=net_dropout,
            num_hops=num_hops
        )
        
        concat_dim = embedding_dim * num_tower
        self.embedding_dim= embedding_dim
        self.num_tower= num_tower
        self.num_fields = num_fields
        self.scorer = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, 1),
        )
        
    def forward(self, x):
        x = x.view(-1, self.num_fields, self.embedding_dim)
        x = self.fusion_netowrk(x)
        return self.scorer(x)