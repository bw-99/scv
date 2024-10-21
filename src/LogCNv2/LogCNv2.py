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
import torch.nn.functional as F

class LogCNv2(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="LogCNv2",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_mask_heads=1,
                 net_dropout=0.1,
                 layer_norm=True,
                 batch_norm=False,
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(LogCNv2, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = MultiHeadFeatureEmbedding(feature_map, embedding_dim * num_heads, num_heads)
        input_dim = feature_map.sum_emb_out_dim()
        print("LogCNv2LogCNv2LogCNv2LogCNv2LogCNv2 input_dim", input_dim)
        self.CN = CrossNetwork(input_dim=input_dim,
                                net_dropout=net_dropout,
                                num_mask_heads=num_mask_heads,
                                layer_norm=layer_norm,
                                batch_norm=batch_norm,
                                num_heads=num_heads)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        logit = self.CN(feature_emb).mean(dim=1)
        y_pred = self.output_activation(logit)
        return_dict = {"y_pred": y_pred}
        return return_dict


class MultiHeadFeatureEmbedding(nn.Module):
    def __init__(self, feature_map, embedding_dim, num_heads=2):
        super(MultiHeadFeatureEmbedding, self).__init__()
        self.num_heads = num_heads
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

    def forward(self, X):  # H = num_heads
        feature_emb = self.embedding_layer(X)  # B × F × D
        multihead_feature_emb = torch.tensor_split(feature_emb, self.num_heads, dim=-1)
        multihead_feature_emb = torch.stack(multihead_feature_emb, dim=1)  # B × H × F × D/H
        multihead_feature_emb1, multihead_feature_emb2 = torch.tensor_split(multihead_feature_emb, 2,
                                                                            dim=-1)  # B × H × F × D/2H
        multihead_feature_emb1, multihead_feature_emb2 = multihead_feature_emb1.flatten(start_dim=2), \
                                                         multihead_feature_emb2.flatten(
                                                             start_dim=2)  # B × H × FD/2H; B × H × FD/2H
        multihead_feature_emb = torch.cat([multihead_feature_emb1, multihead_feature_emb2], dim=-1)
        return multihead_feature_emb  # B × H × FD/H



class CrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 layer_norm=True,
                 batch_norm=True,
                 num_mask_heads=1,
                 net_dropout=0.1,
                 num_heads=1):
        super(CrossNetwork, self).__init__()
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.masker = nn.ModuleList()
        self.projection = nn.ModuleList()

        self.num_mask_heads = num_mask_heads

        self.w.append(nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(self.num_mask_heads)]))
        self.masker.append(nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(self.num_mask_heads)]))
        if layer_norm:
            self.layer_norm.append(nn.LayerNorm(input_dim))
        if batch_norm:
            self.batch_norm.append(nn.BatchNorm1d(num_heads))
        if net_dropout > 0:
            self.dropout.append(nn.Dropout(net_dropout))
        
        self.sfc = nn.Linear(input_dim * num_mask_heads, 1)

    def forward(self, x):
        x_emb = x
        x = None
        x_lst = []
        idx=0
        for i in range(self.num_mask_heads):
            mask = self.masker[idx][i](x_emb)
            tmpx = torch.log(F.relu(x_emb * mask) + 1)
            tmpx = torch.exp(self.w[idx][i](tmpx))
            x_lst.append(tmpx)
        x_emb = torch.cat(x_lst, dim=-1)

        if len(self.batch_norm) > idx:
            x_emb = self.batch_norm[idx](x_emb)

        logit = self.sfc(x_emb)
        return logit