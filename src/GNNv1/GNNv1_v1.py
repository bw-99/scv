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

class GNNv1_v1(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="GNNv1_v1",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0.1,
                 num_hops=2,
                 layer_norm=True,
                 batch_norm=True,
                 pooling_type="mean",
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(GNNv1_v1, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.num_hops = num_hops

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()

        print("num feileds", feature_map.get_num_fields())
        self.num_fields = feature_map.get_num_fields()
        self.input_dim = input_dim
        print("GNNv1_v1GNNv1_v1GNNv1_v1GNNv1_v1GNNv1_v1 input_dim", input_dim)

        self.gnn_tower = CrossNetwork(
                                num_fields=self.num_fields,
                                embedding_dim=embedding_dim,
                                net_dropout=net_dropout,
                                num_hops=num_hops,
                                layer_norm=layer_norm,
                                pooling_type=pooling_type,
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
        # feature_emb = feature_emb.transpose(1, 2) # (bs, feat size, num field)
        # print("feature_emb ", feature_emb.shape)
        output_lst, var_lst = [], []

        gnn_emb = self.gnn_tower(feature_emb)
        gnn_logit = self.scorer(gnn_emb)

        y_pred = self.output_activation(gnn_logit)
        return_dict = {"y_pred": y_pred}
        return return_dict

class CrossNetwork(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 layer_norm=True,
                 batch_norm=True,
                 pooling_type="mean",
                 num_hops=2,
                 net_dropout=0.1):
        super(CrossNetwork, self).__init__()
        self.num_hops=num_hops
        
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = None
        self.w = nn.ParameterList()
        self.masker = nn.ParameterList()

        self.convs = nn.ModuleList()

        # Pooling layers
        self.pool = GlobalPooling(pooling_type)

        for i in range(self.num_hops):
            self.convs.append(SAGEConv(embedding_dim, embedding_dim))
            self.w.append(nn.Parameter(torch.zeros((num_fields, num_fields)), requires_grad=True))
            self.masker.append(nn.Parameter(torch.zeros((num_fields, num_fields)), requires_grad=True))
            
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(embedding_dim))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(embedding_dim))
            if net_dropout > 0:
                self.dropout = nn.Dropout(net_dropout)
            
            nn.init.xavier_uniform_(self.w[i].data)
            nn.init.xavier_uniform_(self.masker[i].data)
        
    def forward(self, x):
        for idx in range(self.num_hops):
            # * adj_matrix: (num_fields, num_fields)
            # * x: (num_fields, embedding_dim)
            adj_matrix = F.relu(self.masker[idx] * self.w[idx])
            x = self.convs[idx](x, adj_matrix)

            if len(self.batch_norm) > idx:
                # x = x.transpose(1, 2)  # (batch_size, hidden_dim, num_nodes)
                x = self.batch_norm[idx](x)
                # x = x.transpose(1, 2)  # (batch_size, num_nodes, hidden_dim) 
            if len(self.layer_norm) > idx:
                x = self.layer_norm[idx](x)

        if self.dropout:
            x = self.dropout(x)

        # * graph embedding pooling
        x_emb = self.pool(x)
        return x_emb