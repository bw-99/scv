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

class LogCNv3(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="LogCNv3",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_mask_heads=1,
                 num_mask_blocks=1,
                 net_dropout=0.1,
                 output_log=False,
                 layer_norm=True,
                 batch_norm=False,
                 exp_positive_activation=False,
                 exp_bias_on_final=False,
                 exp_additional_mask=True,
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(LogCNv3, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = MultiHeadFeatureEmbedding(feature_map, embedding_dim * num_heads, num_heads)
        input_dim = feature_map.sum_emb_out_dim()
        print("LogCNv3LogCNv3LogCNv3LogCNv3LogCNv3 input_dim", input_dim)
        self.num_mask_heads = num_mask_heads
        self.log_tower = nn.ModuleList([CrossNetwork(input_dim=input_dim,
                                net_dropout=net_dropout,
                                num_mask_blocks=num_mask_blocks,
                                layer_norm=layer_norm,
                                output_log=output_log,
                                exp_positive_activation=exp_positive_activation,
                                exp_additional_mask=exp_additional_mask,
                                exp_bias_on_final=exp_bias_on_final,
                                batch_norm=batch_norm,
                                num_heads=num_heads) for _ in range(num_mask_heads)])
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        output_lst = torch.cat([
            self.log_tower[i](feature_emb).mean(dim=1) for i in range(self.num_mask_heads)
        ], dim=-1)
        # print(output_lst.shape)
        # print(torch.mean(output_lst, dim=0).shape, torch.mean(output_lst, dim=1, keepdim=True).shape, torch.mean(output_lst, dim=-1).shape)
        y_pred = torch.mean(output_lst, dim=1,  keepdim=True)
        logit_lst = self.output_activation(output_lst)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred, "logit_lst": logit_lst}
        return return_dict
    
    def train_step(self, inputs):
        self.optimizer.zero_grad()
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        y_pred = return_dict["y_pred"]
        logit_lst = return_dict["logit_lst"]

        loss = self.loss_fn(y_pred, y_true, reduction='mean')
        loss_lst = [self.loss_fn(logit_lst[:, idx].unsqueeze(dim=-1), y_true, reduction='mean') for idx in range(logit_lst.shape[-1])]
        loss_lst = torch.stack(loss_lst, dim=-1)

        weight_lst = loss_lst - loss
        weight_lst = torch.where(weight_lst > 0, weight_lst, torch.zeros(1).to(weight_lst.device))
        additional_loss = loss_lst * weight_lst

        loss = loss + additional_loss.sum()
        loss += self.regularization_loss()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss



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
                 output_log=False,
                 exp_positive_activation=False,
                 exp_additional_mask=True,
                 exp_bias_on_final=False,
                 num_mask_blocks=1,
                 net_dropout=0.1,
                 num_heads=1):
        super(CrossNetwork, self).__init__()
        self.num_mask_blocks = num_mask_blocks
        self.output_log = output_log
        self.exp_bias_on_final = exp_bias_on_final
        self.exp_additional_mask = exp_additional_mask

        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ParameterList()
        self.make_positive = nn.ModuleList()
        self.b = nn.ParameterList()
        self.masker = nn.ParameterList()
        
        for i in range(self.num_mask_blocks):
            self.w.append(nn.Parameter(torch.zeros((input_dim, input_dim)), requires_grad=True))
            self.b.append(nn.Parameter(torch.zeros((input_dim,)), requires_grad=True))
            self.masker.append(nn.Parameter(torch.zeros((input_dim, input_dim)), requires_grad=True))
            if exp_positive_activation:
                self.make_positive.append(nn.Softplus())
            else:
                self.make_positive.append(nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU()
                ))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))

            nn.init.xavier_uniform_(self.w[i].data)
            nn.init.xavier_uniform_(self.masker[i].data)
            nn.init.constant_(self.b[i].data, 0.01)

        
        self.sfc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x_emb = x
        x = None
        for idx in range(self.num_mask_blocks):
            pos_x = self.make_positive[idx](x_emb) + 1
            log_x = torch.log(pos_x)
            if self.exp_additional_mask:
                masked_weight = F.relu(self.masker[idx] * self.w[idx])
            else:
                masked_weight = F.relu(self.w[idx])
            x_emb = log_x @ masked_weight
            if not self.exp_bias_on_final:
                x_emb = x_emb + self.b[idx]

            if len(self.batch_norm) > idx:
                x_emb = self.batch_norm[idx](x_emb)
            if not self.output_log:
                x_emb = torch.exp(x_emb)
            
            if self.exp_bias_on_final:
                x_emb = x_emb + self.b[idx]
            
            # print(idx, x_emb)
        logit = self.sfc(x_emb)
        return logit