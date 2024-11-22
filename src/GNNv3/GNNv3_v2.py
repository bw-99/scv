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

class GNNv3_v2(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="GNNv3_v2",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0.1,
                 num_hops=2,
                 num_mask=2,
                 num_tower=1,
                 use_same_adj=False,
                 nomalize_adj=True,
                 layer_norm=True,
                 batch_norm=True,
                 pooling_dim=2,
                 pooling_type="mean",
                 embedding_regularizer=None,
                 parallel_dnn_hidden_units= [400,400,400],
                 net_regularizer=None,
                 **kwargs):
        super(GNNv3_v2, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.num_hops = num_hops
        self.pooling_dim = pooling_dim
        self.num_tower = num_tower
        self.nomalize_adj = nomalize_adj
        self.num_mask = num_mask
        self.use_same_adj = use_same_adj
        self.gpu = gpu

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()

        print("num feileds", feature_map.get_num_fields())
        self.num_fields = feature_map.get_num_fields()
        self.input_dim = input_dim
        print("GNNv3_v2GNNv3_v2GNNv3_v2GNNv3_v2GNNv3_v2 input_dim", input_dim)

        self.gnn_tower = nn.ModuleList([
            CrossNetwork(
                num_fields=self.num_fields,
                embedding_dim=embedding_dim,
                net_dropout=net_dropout,
                num_hops=num_hops,
                num_mask=self.num_mask,
                layer_norm=layer_norm,
                pooling_type=pooling_type,
                use_same_adj=self.use_same_adj,
                pooling_dim=pooling_dim,
                batch_norm=batch_norm,
                gpu=self.gpu,
                nomalize_adj=self.nomalize_adj
            ) for _ in range(self.num_tower)
        ])

        self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                          output_dim=1, # output hidden layer
                                          hidden_units=parallel_dnn_hidden_units,
                                          hidden_activations="ReLU",
                                          output_activation=None, 
                                          dropout_rates=net_dropout, 
                                          batch_norm=batch_norm)

        final_dim = embedding_dim if pooling_dim==1 else self.num_fields
        self.scorer = nn.Sequential(
            nn.Linear(self.num_tower * final_dim, self.num_tower * final_dim),
            nn.ReLU(),
            nn.Linear(self.num_tower * final_dim, self.num_tower * final_dim),
            nn.ReLU(),
            nn.Linear(self.num_tower * final_dim, self.num_tower * final_dim),
            nn.ReLU(),
            nn.Linear(self.num_tower * final_dim, 1)
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
        flattened_emb = feature_emb.view(feature_emb.shape[0], -1)
        # feature_emb = feature_emb.transpose(1, 2) # (bs, feat size, num field)
        # print("feature_emb ", feature_emb.shape)
        output_lst, var_lst = [], []
        gnn_emb_lst = []
        for idx in range(self.num_tower):
            gnn_emb_lst.append(self.gnn_tower[idx](feature_emb))
        gnn_emb_lst = torch.cat(gnn_emb_lst, dim=-1)
        dnn_emb = self.parallel_dnn(flattened_emb)

        gnn_pred = self.output_activation(self.scorer(gnn_emb_lst))
        deep_pred = self.output_activation(dnn_emb)

        y_pred = (gnn_pred+deep_pred)/2

        return_dict = {"y_pred": y_pred, "gnn_pred": gnn_pred, "deep_pred":deep_pred}
        return return_dict

    def train_step(self, inputs):
        self.optimizer.zero_grad()
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        y_pred = return_dict["y_pred"]
        gnn_pred = return_dict["gnn_pred"]
        deep_pred = return_dict["gnn_pred"]

        loss = self.loss_fn(y_pred, y_true, reduction='mean')
        
        loss = loss + self.loss_fn(gnn_pred, y_pred, reduction='mean') + self.loss_fn(deep_pred, y_pred, reduction='mean')
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
                 use_same_adj=False,
                 num_mask=2,
                 pooling_type="mean",
                 pooling_dim=2,
                 gpu=0,
                 nomalize_adj=True,
                 num_hops=2,
                 net_dropout=0.1):
        super(CrossNetwork, self).__init__()
        self.num_hops=num_hops
        self.nomalize_adj = nomalize_adj

        self.use_same_adj = use_same_adj
        self.num_mask = num_mask
        
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = None
        # self.w = nn.ParameterList()

        self.masker = []

        self.convs = nn.ModuleList()

        # Pooling layers
        self.pool = GlobalPooling(pooling_type, pooling_dim=pooling_dim)

        for i in range(self.num_hops):
            self.convs.append(SAGEConv3(embedding_dim, embedding_dim, nomalize_adj=nomalize_adj))
            
            if self.use_same_adj:
                self.masker = [
                    nn.Parameter(torch.zeros((num_fields, num_fields)).to(f"cuda:{gpu}"), requires_grad=True)
                    for _ in range(self.num_mask)
                ]
                for idx in range(self.num_mask):
                    nn.init.xavier_uniform_(self.masker[idx].data)
            else:
                self.masker.append([
                    nn.Parameter(torch.zeros((num_fields, num_fields)).to(f"cuda:{gpu}"), requires_grad=True)
                    for _ in range(self.num_mask)
                ])
                for idx in range(self.num_mask):
                    nn.init.xavier_uniform_(self.masker[i][idx].data)
            
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(embedding_dim))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(embedding_dim))
            if net_dropout > 0:
                self.dropout = nn.Dropout(net_dropout)
        
        
    def forward(self, x):
        for idx in range(self.num_hops):
            # * adj_matrix: (num_fields, num_fields)
            # * x: (num_fields, embedding_dim)
            if self.use_same_adj:
                adj_matrix = self.masker[0]
                for i in range(1, self.num_mask):
                    adj_matrix = self.masker[i] * adj_matrix
                adj_matrix = F.relu(adj_matrix)
            else:
                adj_matrix = self.masker[idx][0]
                for i in range(1, self.num_mask):
                    adj_matrix = self.masker[idx][i] * adj_matrix
                adj_matrix = F.relu(adj_matrix)

            x = self.convs[idx](x, adj_matrix)

            if len(self.batch_norm) > idx:
                x = self.batch_norm[idx](x)
            if len(self.layer_norm) > idx:
                x = self.layer_norm[idx](x)

        if self.dropout:
            x = self.dropout(x)

        # * graph embedding pooling
        x_emb = self.pool(x)
        return x_emb