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
from tqdm import tqdm
import sys
import logging
import numpy as np
import os

class LogCNv3_exp_var(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="LogCNv3_exp_var",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_mask_heads=1,
                 num_mask_blocks=1,
                 net_dropout=0.1,
                 output_log=False,
                 layer_norm=True,
                 batch_norm=True,
                 exp_positive_activation=False,
                 exp_bias_on_final=False,
                 exp_additional_mask=True,
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 expid=None,
                 **kwargs):
        super(LogCNv3_exp_var, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.num_mask_blocks = num_mask_blocks
        self.num_mask_heads = num_mask_heads
        self.expid = expid
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        print("LogCNv3_exp_varLogCNv3_exp_varLogCNv3_exp_varLogCNv3_exp_varLogCNv3_exp_var input_dim", input_dim)
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
        self.tmp_val_lst, self.tmp_after_val_lst, self.tmp_before_val_lst = [], [], []
        self.val_lst = []

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        output_lst, var_lst, before_log_var_lst, after_log_var_lst = [], [], [], []
        for i in range(self.num_mask_heads):
            logit, x_emb, before_log_var, after_log_var = self.log_tower[i](feature_emb)
            feature_variances = torch.var(x_emb, dim=0)
            var_lst.append(feature_variances.mean().detach().cpu())
            output_lst.append(logit)
            
            before_log_var_lst.append(before_log_var)
            after_log_var_lst.append(after_log_var)

        output_lst = torch.cat(output_lst, dim=-1)
        before_log_var_lst = torch.cat(before_log_var_lst, dim=-1).mean(dim=1)
        after_log_var_lst = torch.cat(after_log_var_lst, dim=-1).mean(dim=1)

        self.tmp_val_lst.append((sum(var_lst)/len(var_lst)).item())
        self.tmp_after_val_lst.append(after_log_var_lst.mean(dim=0).item())
        self.tmp_before_val_lst.append(before_log_var_lst.mean(dim=0).item())

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
            # print(sum(self.tmp_val_lst)/len(self.tmp_val_lst))
            train_loss += loss.item()
            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break
        
        os.makedirs(f"analysis/{self.expid}", exist_ok=True)

        np.save(f"analysis/{self.expid}/final_emb_var.npy", np.array(self.tmp_val_lst))
        np.save(f"analysis/{self.expid}/before_log_emb_var.npy", np.array(self.tmp_before_val_lst))
        np.save(f"analysis/{self.expid}/after_log_emb_var.npy", np.array(self.tmp_after_val_lst))

        self.tmp_val_lst.clear()
        self.tmp_before_val_lst.clear()
        self.tmp_after_val_lst.clear()




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
                 num_mask_blocks=1,
                 net_dropout=0.1,
                 num_heads=1):
        super(CrossNetwork, self).__init__()
        self.num_mask_blocks = num_mask_blocks
        self.output_log = output_log
        self.exp_bias_on_final = exp_bias_on_final
        self.exp_additional_mask = exp_additional_mask
        self.exp_add1_before_log = exp_add1_before_log

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
            nn.init.constant_(self.b[i].data, 0.01)

        
        self.sfc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x_emb = x
        x = None
        before_log_var, after_log_var = [], []
        for idx in range(self.num_mask_blocks):
            pos_x = self.make_positive[idx](x_emb)
            if self.exp_add1_before_log:
                pos_x=pos_x+1
            before_log_var.append(torch.var(pos_x, dim=0).mean())
            log_x = torch.log(pos_x)
            after_log_var.append(torch.var(log_x, dim=0).mean())
            if self.exp_additional_mask:
                masked_weight = F.relu(self.masker[idx] * self.w[idx])
            else:
                masked_weight = F.relu(self.w[idx])
            x_emb = log_x @ masked_weight

            if not self.exp_bias_on_final:
                x_emb = x_emb + self.b[idx]

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

        logit = self.sfc(x_emb)
        # print(len(before_log_var), before_log_var[0].shape)
        # print(torch.stack(before_log_var).shape)
        # print(torch.stack(before_log_var).unsqueeze(dim=1).mean().detach().cpu())
        return logit, x_emb, torch.stack(before_log_var).unsqueeze(dim=1).detach().cpu(), torch.stack(after_log_var).unsqueeze(dim=1).detach().cpu()