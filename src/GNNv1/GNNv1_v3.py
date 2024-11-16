# =========================================================================
# Copyright (C) 2024 salmon@github
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_regularizer
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetV2, CrossNetMix
import torch.nn.functional as F
from .util import *

class GNNv1_v3(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="GNNv1_v3",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0.1,
                lambda_pool=1.0,
                 num_hops=2,
                 layer_norm=True,
                 batch_norm=True,
                 num_clusters=4,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(GNNv1_v3, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.num_hops = num_hops
        self.lambda_pool = lambda_pool

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()

        print("num feileds", feature_map.get_num_fields())
        self.num_fields = feature_map.get_num_fields()
        self.input_dim = input_dim
        print("GNNv1_v3GNNv1_v3GNNv1_v3GNNv1_v3GNNv1_v3 input_dim", input_dim)

        self.gnn_tower = CrossNetwork(
                                num_fields=self.num_fields,
                                embedding_dim=embedding_dim,
                                net_dropout=net_dropout,
                                num_hops=num_hops,
                                num_clusters=num_clusters,
                                layer_norm=layer_norm,
                                batch_norm=batch_norm)
        
        final_dim = embedding_dim
        self.scorer = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, 1)
        )

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        self.tmp_val_lst = []
        self.val_lst = []

        print("without emb dim")
        self.count_parameters(count_embedding=False)

        

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=False)

        gnn_emb, pool_loss = self.gnn_tower(feature_emb)
        gnn_logit = self.scorer(gnn_emb)

        y_pred = self.output_activation(gnn_logit)
        return_dict = {"y_pred": y_pred, "pool_loss": self.lambda_pool * pool_loss}
        return return_dict
    
    def train_step(self, inputs):
        self.optimizer.zero_grad()
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        
        y_pred = return_dict["y_pred"]
        pool_loss = return_dict["pool_loss"]

        loss = self.loss_fn(y_pred, y_true, reduction='mean') + pool_loss

        loss += self.regularization_loss()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss

class CrossNetwork(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 layer_norm=True,
                 batch_norm=True,
                 num_hops=2,
                 num_clusters=4,
                 net_dropout=0.1):
        super(CrossNetwork, self).__init__()
        self.num_hops=num_hops
        
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ParameterList()
        self.masker = nn.ParameterList()

        self.convs = nn.ModuleList()

        self.pool = DiffPool(embedding_dim, num_clusters)

        self.meta_adj_matrix = nn.Parameter(torch.zeros((num_fields, num_fields)), requires_grad=True)
        self.meta_adj_matrix2 = nn.Parameter(torch.zeros((num_fields, num_fields)), requires_grad=True)

        for i in range(self.num_hops):
            self.convs.append(SAGEConv(embedding_dim, embedding_dim))
            self.w.append(nn.Parameter(torch.zeros((num_fields, num_fields)), requires_grad=True))
            self.masker.append(nn.Parameter(torch.zeros((num_fields, num_fields)), requires_grad=True))
            
            # if layer_norm:
            #     self.layer_norm.append(nn.LayerNorm(input_dim))
            # if batch_norm:
            #     self.batch_norm.append(nn.BatchNorm1d(input_dim))
            # if net_dropout > 0:
            #     self.dropout.append(nn.Dropout(net_dropout))
            
            nn.init.xavier_uniform_(self.w[i].data)
            nn.init.xavier_uniform_(self.masker[i].data)

    def normalize_adj(self, adj):
        degree = adj.sum(dim=-1)
        degree = torch.clamp(degree, min=1e-12)
        degree_inv = 1.0 / degree
        adj_norm = degree_inv.unsqueeze(-1) * adj
        return adj_norm
        
    def forward(self, x):
        for idx in range(self.num_hops):
            # * adj_matrix: (num_fields, num_fields)
            # * x: (num_fields, embedding_dim)
            adj_matrix = F.relu(self.masker[idx] * self.w[idx])
            x = self.convs[idx](x, adj_matrix)

        # * graph embedding pooling
        adj_norm = self.normalize_adj(F.relu(self.meta_adj_matrix * self.meta_adj_matrix2))

        x, adj_norm, link_loss, ent_loss = self.pool(x, adj_norm)
        x = torch.mean(x, dim=1)  # Global pooling after DiffPool
        pool_loss = link_loss + ent_loss

        return x, pool_loss


