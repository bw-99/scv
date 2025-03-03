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


class CrossNetwork(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 layer_norm=True,
                 batch_norm=True,
                 num_tower=2,
                 pooling_method="attn",
                 net_dropout=0.1,
                 num_mask=3,
                 num_hops=1):
        super(CrossNetwork, self).__init__()
        self.num_tower = num_tower
        self.layer_norm_flag = layer_norm
        self.batch_norm_flag = batch_norm
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.pooling_method=pooling_method
        self.num_hops = num_hops
        self.num_mask = num_mask

        self.gnn_transform_weight = Param(torch.Tensor(1, num_tower, num_hops, 2*embedding_dim, embedding_dim))
        self.gnn_transform_bias = Param(torch.Tensor(num_tower, num_hops, num_fields, embedding_dim))
        nn.init.xavier_uniform_(self.gnn_transform_weight)
        nn.init.xavier_uniform_(self.gnn_transform_bias)

        self.masker = nn.Parameter(torch.zeros((self.num_tower, self.num_mask, num_fields, num_fields)))
        nn.init.xavier_uniform_(self.masker)
        self.diag_adj = nn.Parameter(torch.eye(self.num_fields, self.num_fields).unsqueeze(dim=0), requires_grad=False)
        
        self.layer_norm = nn.LayerNorm(num_fields)
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        for i in range(num_hops):
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(self.num_tower * self.num_fields * 2 * self.embedding_dim))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
        
        if self.pooling_method == "attn":
            self.gating_network = nn.Sequential(
                nn.Linear(embedding_dim * num_fields, num_fields),
                nn.Softmax(dim=2)
            )


    def forward(self, x):
        """
        x: (batch_size, num_fields, embedding_dim)
        Output: (batch_size, num_tower * embedding_dim)
        """
        # Expand x for multiple towers => (B, T, N, D)
        x = x.unsqueeze(1).repeat(1, self.num_tower, 1, 1)

        # shape => (num_tower, num_hops, num_mask, N, N)
        adj_matrix_hop = torch.prod(self.masker, dim=1) # (num_tower, num_fields, num_fields)
        adj_matrix_hop = F.relu(adj_matrix_hop)
        mask = (adj_matrix_hop != 0).float()
        adj_matrix_hop = self.layer_norm(adj_matrix_hop.transpose(-2, -1)).transpose(-2, -1)
        x_masked = adj_matrix_hop + (1 - mask) * -1e9 + self.diag_adj
        adj_matrix_hop = torch.nn.functional.softmax(x_masked, dim=1) * mask

        for idx in range(self.num_hops):
            # 1) Linear transform => (B, T, N, D)
            nei_features = torch.matmul(x.transpose(-2, -1), adj_matrix_hop).transpose(-2, -1)
            nei_features = nei_features
            
            transform = self.gnn_transform_weight[:, :, idx, ...]
            x = torch.cat([x, nei_features], dim=-1) # B, T, N, 2D
            
            if len(self.batch_norm) > idx:
                x_reshaped = x.view(x.size(0), -1)  # => (B, T*N*D)
                x_reshaped = self.batch_norm[idx](x_reshaped)
                x = x_reshaped.view(x.size(0), self.num_tower, self.num_fields, -1)
                
            x = torch.matmul(x, transform) + self.gnn_transform_bias[:, idx, ...]
            if len(self.dropout) > idx:
                x = self.dropout[idx](x)   

        if self.pooling_method == "attn":
            weights = self.gating_network(x.view(x.shape[0], x.shape[1], -1)) # B, T, N*D -> B, T, N
            x = torch.sum(x * weights.unsqueeze(dim=-1), dim=2)
        else:
            x = x.mean(dim=2)
        return x


class GNNv3_v14(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="GNNv3_v14",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0.1,
                 num_tower=2,
                 num_hops=1,
                 num_mask=3,
                 pooling_method="attn",
                 use_bilinear_fusion=False,
                 parallel_dnn_hidden_units= [400,400,400],
                 layer_norm=False,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(GNNv3_v14, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.num_tower = num_tower
        self.num_tower = num_tower
        self.gpu = gpu
        self.num_hops = num_hops
        self.use_bilinear_fusion = use_bilinear_fusion
        self.num_mask = num_mask

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()

        print("num fields", feature_map.get_num_fields())
        self.num_fields = feature_map.get_num_fields()
        self.input_dim = input_dim
        print("GNNv3_v14 input_dim", input_dim)

        self.gnn_tower = CrossNetwork(
            num_fields=self.num_fields,
            embedding_dim=embedding_dim,
            net_dropout=net_dropout,
            num_tower=num_tower,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            pooling_method=pooling_method,
            num_hops=num_hops,
            num_mask=num_mask,
        )

        self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                      output_dim=None,  # output hidden layer
                                      hidden_units=parallel_dnn_hidden_units,
                                      hidden_activations="ReLU",
                                      output_activation=None,
                                      dropout_rates=net_dropout,
                                      batch_norm=batch_norm)

        
        if self.use_bilinear_fusion:
            concat_dim = embedding_dim * num_tower
            print("concat_dim ", concat_dim)
            # self.bias = nn.Parameter(torch.tensor(0.0))
            # self.w1 = nn.Parameter(torch.randn(concat_dim, 1))  # Shape: (d1, 1)
            # self.w2 = nn.Parameter(torch.randn(parallel_dnn_hidden_units[-1], 1))  # Shape: (d2, 1)
            # self.W3 = nn.Parameter(torch.randn(concat_dim, parallel_dnn_hidden_units[-1]))  # Shape: (d1, d2)
            # nn.init.xavier_uniform_(self.w1)
            # nn.init.xavier_uniform_(self.w2)
            # nn.init.xavier_uniform_(self.W3)

            self.attention = nn.Linear(dim1 + dim2, 1)  # Attention score 계산
            self.bilinear = nn.Bilinear(dim1, dim2, output_dim)  # Bilinear fusion
        else:
            concat_dim = embedding_dim * num_tower + parallel_dnn_hidden_units[-1]
            print("concat_dim ", concat_dim)
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

        # B, T, D
        graph_embeddings = self.gnn_tower(feature_emb).view(feature_emb.shape[0], -1)
        mlp_embeddings = self.parallel_dnn(feature_emb.view(feature_emb.shape[0], -1))

        if self.use_bilinear_fusion:
            linear_term1 = graph_embeddings @ self.w1  # (B, output_dim)
            linear_term2 = mlp_embeddings @ self.w2  # (B, output_dim)
            bilinear_term = torch.einsum('bi,ij,bj->b', graph_embeddings, self.W3, mlp_embeddings).unsqueeze(1)  # (B, 1)
            y_pred = self.bias + linear_term1 + linear_term2 + bilinear_term
        else:
            y_pred = self.scorer(torch.cat([graph_embeddings, mlp_embeddings], dim=-1))

        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict