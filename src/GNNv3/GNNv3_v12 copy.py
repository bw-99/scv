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
                 net_dropout=0.1,
                 num_mask=3,
                 num_hops=1):
        super(CrossNetwork, self).__init__()
        self.num_tower = num_tower
        self.layer_norm_flag = layer_norm
        self.batch_norm_flag = batch_norm
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
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
        
        self.gating_network = nn.Sequential(
            nn.Linear(embedding_dim * num_fields, 1),
            nn.Softmax(dim=1)
        )

    def modify_adjacency_matrix(self, adj_matrix, edge_pertub_prob):
        # 기존 Adjacency Matrix 복사
        adj = adj_matrix.clone()

        # 새 간선 추가 (값이 0인 곳에만 추가)
        zero_mask = (adj == 0).float()  # 값이 0인 곳에 1
        add_mask = torch.bernoulli(edge_pertub_prob * zero_mask)  # 0.1 확률로 간선 추가
        adj += add_mask  # 새 간선 추가

        # 기존 간선 삭제 (값이 1인 곳에서만 삭제)
        one_mask = (adj > 0).float()  # 값이 1인 곳에 1
        drop_mask = torch.bernoulli(edge_pertub_prob * one_mask)  # 0.1 확률로 간선 삭제
        adj -= drop_mask * 1e8  # 기존 간선 제거
        adj = F.relu(adj)

        return adj


    def forward(self, x, edge_pertub_prob=0):
        """
        x: (batch_size, num_fields, embedding_dim)
        Output: (batch_size, num_tower * embedding_dim)
        """
        # Expand x for multiple towers => (B, T, N, D)
        x = x.unsqueeze(1).repeat(1, self.num_tower, 1, 1)

        # shape => (num_tower, num_hops, num_mask, N, N)
        adj_matrix_hop = torch.prod(self.masker, dim=1) # (num_tower, num_fields, num_fields)
        adj_matrix_hop = F.relu(adj_matrix_hop)
        if edge_pertub_prob > 0:
            adj_matrix_hop = self.modify_adjacency_matrix(adj_matrix_hop, edge_pertub_prob)
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

            x = x.view(x.shape[0], x.shape[1], -1) # B, T, N*D

        weights = self.gating_network(x) # B, T, N*D -> B, T, 1
        x = torch.sum(x * weights, dim=1)
        return x


class GNNv3_v12(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="GNNv3_v12",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0.1,
                 num_tower=2,
                 num_hops=1,
                 num_mask=3,
                 edge_pertub_prob=0,
                 layer_norm=False,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(GNNv3_v12, self).__init__(feature_map,
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
        self.num_mask = num_mask
        self.edge_pertub_prob = edge_pertub_prob

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()

        print("num fields", feature_map.get_num_fields())
        self.num_fields = feature_map.get_num_fields()
        self.input_dim = input_dim
        print("GNNv3_v12 input_dim", input_dim)

        self.gnn_tower = CrossNetwork(
            num_fields=self.num_fields,
            embedding_dim=embedding_dim,
            net_dropout=net_dropout,
            num_tower=num_tower,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            num_hops=num_hops,
            num_mask=num_mask,
        )

        concat_dim = embedding_dim * self.num_fields
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

    def forward(self, inputs, p=0):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=False)

        # B, T, D
        graph_embeddings = self.gnn_tower(feature_emb, edge_pertub_prob=p)

        y_pred = self.scorer(graph_embeddings.view(feature_emb.shape[0], -1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred, "graph_embedding": graph_embeddings}
        return return_dict
    
    def train_step(self, inputs):
        self.optimizer.zero_grad()
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        
        y_pred = return_dict["y_pred"]
        loss = self.loss_fn(y_pred, y_true, reduction='mean')

        if self.edge_pertub_prob > 0:
            temperature = 0.5
            H = return_dict["graph_embedding"]
            B, T, D = H.size()
            noisy_result_dict = self.forward(inputs, p=self.edge_pertub_prob)
            H_prime = noisy_result_dict["graph_embedding"]

            # Flattened 임베딩
            H_flat = H.view(B * T, D)  # (B*T, D)
            H_prime_flat = H_prime.view(B * T, D)  # (B*T, D)

            # * Positive Pair 유사도 계산 (대각선)
            pos_sim = F.cosine_similarity(H_flat, H_prime_flat, dim=1)  # (B*T,)
            exp_pos_sim = torch.exp(pos_sim / temperature).view(B, T, 1)  # (B, T, 1)

            # * Hard Negative Pair
            negative_indices_per_t = []
            for t in range(T):
                idx = [tt for tt in range(T) if tt != t]
                negative_indices_per_t.append(idx)
            negative_indices_per_t = torch.tensor(negative_indices_per_t, device=H.device)  # (T, T-1)
            negative_indices = negative_indices_per_t.unsqueeze(0).expand(B, T, T-1)  # (B, T, T-1)
            b_range = torch.arange(B, device=H.device).view(B, 1, 1).expand(B, T, T-1)  # (B, T, T-1)
            hard_negatives = H[b_range, negative_indices, :]  # (B, T, T-1, D)

            hard_neg_sim = F.cosine_similarity(
                H.unsqueeze(2), hard_negatives, dim=-1
            )  # (B, T, T-1)
            exp_hardneg_sim = torch.exp(hard_neg_sim / temperature).sum(dim=2, keepdim=True)  # (B, T, 1)

            # # * In-batch Negative Sampling
            # num_negatives = 1  # 샘플링할 negative 개수
            # num_samples = H_flat.size(0)  # B*T
            # negative_indices = torch.randint(0, num_samples, (num_samples, num_negatives), device=H_flat.device)  # (B*T, 100)
            # negative_samples = H_prime_flat[negative_indices]  # (B*T, 100, D)
            # negative_sim = F.cosine_similarity(
            #     H_flat.unsqueeze(1), negative_samples, dim=2
            # )  # (B*T, 100)
            # exp_neg_sim = torch.exp(negative_sim / temperature).sum(dim=1, keepdim=True).view(B, T, 1)  # (B, T, 1)

            # * InfoNCE Loss 계산
            # denominator = exp_pos_sim + exp_hardneg_sim + exp_neg_sim  # (B, T, 1)
            denominator = exp_pos_sim + exp_hardneg_sim  # (B, T, 1)
            infonce_loss = -torch.log(exp_pos_sim / denominator)  # (B, T, 1)
            loss += infonce_loss.mean() 
            loss += F.mse_loss(noisy_result_dict["y_pred"], y_pred, reduction='mean')

        loss += self.regularization_loss()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss