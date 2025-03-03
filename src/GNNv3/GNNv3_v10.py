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
                 use_gru=True,
                 pooling_type="mean",
                 pooling_dim=2,
                 num_tower=2,
                 net_dropout=0.1,
                 num_hops=1):
        super(CrossNetwork, self).__init__()
        self.num_tower = num_tower
        self.layer_norm_flag = layer_norm
        self.batch_norm_flag = batch_norm
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.num_hops = num_hops
        self.use_gru = use_gru


        # self.masker_1st_param = nn.Parameter(torch.ones(num_tower, num_fields*embedding_dim, num_fields*embedding_dim))
        # self.masker_1st_bias = nn.Parameter(torch.ones(num_tower, num_fields*embedding_dim))
        self.masker_2nd_param = nn.Parameter(torch.ones(num_tower, num_fields*embedding_dim, num_fields*num_fields))
        self.masker_2nd_bias = nn.Parameter(torch.ones(num_tower, num_fields*num_fields))
        
        # nn.init.xavier_uniform_(self.masker_1st_param)
        # nn.init.xavier_uniform_(self.masker_1st_bias)
        nn.init.xavier_uniform_(self.masker_2nd_param)
        nn.init.xavier_uniform_(self.masker_2nd_bias)

        # Weights for each hop
        # shape = (num_hops, embedding_dim, embedding_dim)
        self.weight = Param(torch.Tensor(num_hops, embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # A single GRUCell shared across all hops
        if self.use_gru:
            self.rnn = MyGRUCell(embedding_dim, embedding_dim, bias=True)
        else:
            self.agg_lin = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim)
            for _ in range(num_hops)])

        if layer_norm:
            self.layer_norm = nn.LayerNorm([self.num_tower, num_fields, embedding_dim])

        if self.batch_norm_flag:
            out_dim = self.num_tower * self.num_fields * self.embedding_dim
            self.batch_norm = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(net_dropout) if net_dropout > 0 else None

        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        """
        x: (batch_size, num_fields, embedding_dim)
        Output: (batch_size, num_tower * embedding_dim)
        """
        batch_size, num_fields, embedding_dim = x.size()
        fusion_x = x.view(batch_size, -1)
        fusion_x = fusion_x.unsqueeze(1).repeat(1, self.num_tower, 1) # (B, T, N*D)
        
        adj_matrix = torch.einsum('btn,tnm->btm', fusion_x, self.masker_2nd_param)
        adj_matrix = adj_matrix + self.masker_2nd_bias
        adj_matrix = F.relu(adj_matrix)
        adj_matrix = adj_matrix.view(batch_size, self.num_tower, num_fields, num_fields)
        adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + 1e-6)

        # Expand x for multiple towers => (B, T, N, D)
        x = x.unsqueeze(1).repeat(1, self.num_tower, 1, 1)

        if self.use_gru:
            for idx in range(self.num_hops):
                m = torch.matmul(x, self.weight[idx])  # still (B, T, N, D)
                m = adj_matrix @ m
                B, T, N, D = m.shape
                m_2d = m.view(-1, D)
                x_2d = x.view(-1, D)
                x_2d = self.rnn(m_2d, x_2d)
                x = x_2d.view(B, T, N, D)
        else:
            for idx in range(self.num_hops):
                m = torch.matmul(x, self.weight[idx])  # still (B, T, N, D)
                x = self.agg_lin[idx](adj_matrix @ m)

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
        B, T, N, D = x.size()
        
        # Step 1: Reshape for linear layer
        x_reshaped = x.view(B * T, N, D)  # (B*T, N, D)
        
        # Step 2: Compute attention scores
        scores = self.attention(x_reshaped)  # (B*T, N, 1)
        scores = scores.squeeze(-1)  # (B*T, N)
        
        # Step 3: Apply softmax to get normalized weights
        weights = F.softmax(scores, dim=1)  # (B*T, N)
        
        # Step 4: Reshape weights and apply them to input
        weights = weights.unsqueeze(-1)  # (B*T, N, 1)
        weighted_sum = torch.sum(weights * x_reshaped, dim=1)  # (B*T, D)
        
        x = weighted_sum.view(batch_size, self.num_tower * embedding_dim) # (B, T, D)
        return x


class GNNv3_v10(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="GNNv3_v10",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0.1,
                 num_tower=2,
                 num_hops=1,
                 use_gru=False,
                 layer_norm=False,
                 batch_norm=False,
                 pooling_dim=2,
                 pooling_type="mean",
                 fusion_type="MLP",
                 embedding_regularizer=None,
                 parallel_dnn_hidden_units=[400,400,400],
                 net_regularizer=None,
                 **kwargs):
        super(GNNv3_v10, self).__init__(feature_map,
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
        self.use_gru = use_gru
        self.gpu = gpu
        self.num_hops = num_hops

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()

        print("num fields", feature_map.get_num_fields())
        self.num_fields = feature_map.get_num_fields()
        self.input_dim = input_dim
        print("GNNv3_v10 input_dim", input_dim)

        self.gnn_tower = CrossNetwork(
            num_fields=self.num_fields,
            embedding_dim=embedding_dim,
            net_dropout=net_dropout,
            num_tower=num_tower,
            layer_norm=layer_norm,
            pooling_type=pooling_type,
            pooling_dim=pooling_dim,
            batch_norm=batch_norm,
            use_gru=use_gru,
            num_hops=num_hops
        )
        print("parallel_dnn_hidden_units ", parallel_dnn_hidden_units)
        final_dim = embedding_dim

        self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                    #   output_dim=final_dim,
                                      hidden_units=parallel_dnn_hidden_units,
                                      hidden_activations="ReLU",
                                      output_activation=None,
                                      dropout_rates=net_dropout,
                                      batch_norm=batch_norm)

        # Combine GNN output + parallel DNN => (num_tower + 1) heads
        concat_dim = embedding_dim * num_tower + parallel_dnn_hidden_units[-1]

        self.scorer = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, 1)
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
        dnn_emb = self.parallel_dnn(flattened_emb)       # (B, embedding_dim)

        # Merge => (B, (num_tower+1)*embedding_dim)
        final_emb = torch.cat([gnn_emb, dnn_emb], dim=-1)
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
        self.attn_layer = nn.Linear(embedding_dim, 1)

        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Final fusion
        self.fusion_network = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, num_fields * embedding_dim)
        B = x.shape[0]
        x = x.view(B, self.num_fields, self.embedding_dim)

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
                 pooling_type="mean",
                 pooling_dim=2,
                 use_gru=False,
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
            pooling_type=pooling_type,
            pooling_dim=pooling_dim,
            use_gru=use_gru,
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