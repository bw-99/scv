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
from tqdm import tqdm
import sys
import logging
import numpy as np

class LogCNv6_v3(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="LogCNv6_v3",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_log_layers=1,
                 num_shallow_layers=4,
                 net_dropout=0.1,
                 output_log=False,
                 layer_norm=True,
                 batch_norm=True,
                 exp_norm_before_log=False,
                 exp_positive_activation=False,
                 exp_bias_on_final=False,
                 exp_layer_norm_before_concat=True,
                 parallel_dnn_hidden_units=[400, 400, 400],
                 exp_additional_mask=True,
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(LogCNv6_v3, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.num_log_layers = num_log_layers
        self.exp_norm_before_log = exp_norm_before_log
        self.num_shallow_layers = num_shallow_layers

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        print("LogCNv6_v3LogCNv6_v3LogCNv6_v3LogCNv6_v3LogCNv6_v3 input_dim", input_dim)

        self.log_tower = CrossNetwork(input_dim=input_dim,
                                net_dropout=net_dropout,
                                num_log_layers=num_log_layers,
                                layer_norm=layer_norm,
                                output_log=output_log,
                                exp_positive_activation=exp_positive_activation,
                                exp_additional_mask=exp_additional_mask,
                                exp_bias_on_final=exp_bias_on_final,
                                exp_norm_before_log=exp_norm_before_log,
                                batch_norm=batch_norm)
        self.shallow_tower = CrossNetV2(
            input_dim=input_dim,
            num_layers=num_shallow_layers,
        )
        self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                    output_dim=None, # output hidden layer
                                    hidden_units=parallel_dnn_hidden_units,
                                    hidden_activations="ReLU",
                                    output_activation=None, 
                                    dropout_rates=net_dropout, 
                                    batch_norm=batch_norm)
        final_dim = parallel_dnn_hidden_units[-1] + input_dim*2
        
        self.scorer = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, 1)
        )

        self.exp_layer_norm_before_concat = exp_layer_norm_before_concat
        if exp_layer_norm_before_concat:
            self.log_layer_norm = nn.LayerNorm(input_dim)
            self.shal_layer_norm = nn.LayerNorm(input_dim)
            self.mlp_layer_norm = nn.LayerNorm(parallel_dnn_hidden_units[-1])

        self.log_scorer = nn.Linear(input_dim, 1)
        self.shallow_scorer = nn.Linear(input_dim, 1)
        self.mlp_scorer = nn.Linear(parallel_dnn_hidden_units[-1], 1)
        
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        self.tmp_val_lst = []
        self.val_lst = []

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        output_lst, var_lst = [], []

        log_emb = self.log_tower(feature_emb)
        shal_emb = self.shallow_tower(feature_emb)
        mlp_emb = self.parallel_dnn(feature_emb)

        log_logit = self.log_scorer(log_emb)
        shal_logit = self.shallow_scorer(shal_emb)
        mlp_logit = self.mlp_scorer(mlp_emb)

        if self.exp_layer_norm_before_concat:
            log_emb = self.log_layer_norm(log_emb)
            shal_emb = self.shal_layer_norm(shal_emb)
            mlp_emb = self.mlp_layer_norm(mlp_emb)

        logit_lst = self.output_activation(torch.cat([log_logit, shal_logit, mlp_logit], dim=-1))
        output_lst = torch.cat([log_emb, shal_emb, mlp_emb], dim=-1)
        y_pred = self.scorer(output_lst)
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

        kd_loss_lst = [self.loss_fn(logit_lst[:, idx].unsqueeze(dim=-1), y_pred, reduction='mean') for idx in range(logit_lst.shape[-1])]
        kd_loss_lst = torch.stack(kd_loss_lst, dim=-1)

        loss += kd_loss_lst.sum()
        loss += self.regularization_loss()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss
    
    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            train_loss += loss.item()
            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break
        self.tmp_val_lst.clear()

class CrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 layer_norm=True,
                 batch_norm=True,
                 output_log=False,
                 exp_positive_activation=False,
                 exp_additional_mask=True,
                 exp_bias_on_final=False,
                 exp_add1_before_log=True,
                 exp_norm_before_log=False,
                 num_log_layers=1,
                 net_dropout=0.1):
        super(CrossNetwork, self).__init__()
        self.num_log_layers = num_log_layers
        self.output_log = output_log
        self.exp_bias_on_final = exp_bias_on_final
        self.exp_additional_mask = exp_additional_mask
        self.exp_add1_before_log = exp_add1_before_log
        self.exp_norm_before_log = exp_norm_before_log

        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ParameterList()
        self.make_positive = nn.ModuleList()
        self.b = nn.ParameterList()
        self.masker = nn.ParameterList()
        
        for i in range(self.num_log_layers):
            self.w.append(nn.Parameter(torch.zeros((input_dim, input_dim)), requires_grad=True))
            self.b.append(nn.Parameter(torch.zeros((input_dim,)), requires_grad=True))
            self.masker.append(nn.Parameter(torch.zeros((input_dim, input_dim)), requires_grad=True))
            if exp_positive_activation:
                self.make_positive.append(nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.Softplus()
                ))
            else:
                self.make_positive.append(nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU()
                ))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(input_dim))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            
            nn.init.xavier_uniform_(self.w[i].data)
            nn.init.xavier_uniform_(self.masker[i].data)
            nn.init.uniform_(self.b[i].data, 0.01)
        
    def forward(self, x):
        x_emb = x
        x = None
        for idx in range(self.num_log_layers):
            if self.exp_norm_before_log:
                if len(self.batch_norm) > idx:
                    x_emb = self.batch_norm[idx](x_emb)
                if len(self.layer_norm) > idx:
                    x_emb = self.layer_norm[idx](x_emb)

            pos_x = self.make_positive[idx](x_emb)
            if self.exp_add1_before_log:
                pos_x=pos_x+1
            
            log_x = torch.log(pos_x)
            if self.exp_additional_mask:
                masked_weight = F.relu(self.masker[idx] * self.w[idx])
            else:
                masked_weight = F.relu(self.w[idx])
            x_emb = log_x @ masked_weight

            if not self.exp_bias_on_final:
                x_emb = x_emb + self.b[idx]

            if not self.exp_norm_before_log:
                if len(self.batch_norm) > idx:
                    x_emb = self.batch_norm[idx](x_emb)
                if len(self.layer_norm) > idx:
                    x_emb = self.layer_norm[idx](x_emb)
            
            if not self.output_log:
                x_emb = torch.exp(x_emb)
            
            if self.exp_bias_on_final:
                x_emb = x_emb + self.b[idx]

            if len(self.dropout) > idx:
                x_emb = self.dropout[idx](x_emb)

        return x_emb