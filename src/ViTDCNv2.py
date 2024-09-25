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
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block



class ViTDCNv2(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="ViTDCNv2",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_cross_layers=4,
                 net_dropout=0.1,
                 layer_norm=True,
                 batch_norm=False,
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 parallel_dnn_hidden_units=[],
                 vit_patch_size=8,
                 vit_hidden_dim=32, 
                 vit_num_layers=2, 
                 vit_num_heads=8,
                 **kwargs):
        super(ViTDCNv2, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = MultiHeadFeatureEmbedding(feature_map, embedding_dim * num_heads, num_heads)
        input_dim = feature_map.sum_emb_out_dim()
        self.crossnet = ViTCrossNetV2(input_dim, num_cross_layers, net_dropout=net_dropout,
                                           layer_norm=layer_norm,
                                           batch_norm=batch_norm,
                                           num_heads=num_heads,
                                           vit_input_dim=input_dim,
                                           vit_patch_size=vit_patch_size,
                                            vit_hidden_dim=vit_hidden_dim,
                                            vit_num_layers=vit_num_layers,
                                            vit_num_heads=vit_num_heads)
        self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                          output_dim=None, # output hidden layer
                                          hidden_units=parallel_dnn_hidden_units,
                                          hidden_activations="ReLU",
                                          output_activation=None, 
                                          dropout_rates=net_dropout, 
                                          batch_norm=batch_norm)

        final_dim = input_dim + parallel_dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        cross_out = self.crossnet(feature_emb)
        dnn_out = self.parallel_dnn(feature_emb)
        final_out = torch.cat([cross_out, dnn_out], dim=-1)
        y_pred = self.fc(final_out)
        if(len(y_pred.shape) == 3):
            y_pred =  y_pred.squeeze(-1)
        y_pred = self.output_activation(y_pred)
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

class ViTCrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers,
                layer_norm=True,
                 batch_norm=False,
                 vit_input_dim=100, 
                 vit_patch_size=16,
                 vit_hidden_dim=768, 
                 vit_num_layers=2, 
                 vit_num_heads=4,
                 net_dropout=0.1,
                 num_heads=1,):
        super(ViTCrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        self.masker_lst = nn.ModuleList()
        for i in range(num_layers):
            self.w.append(nn.Linear(input_dim, input_dim, bias=False))
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            self.masker_lst.append(nn.Sequential(
                VIT2DEmbeddingModel(
                    vit_input_dim=vit_input_dim,
                    vit_patch_size=vit_patch_size,
                    vit_hidden_dim=vit_hidden_dim,
                    vit_num_layers=vit_num_layers,
                    vit_num_heads=vit_num_heads
                ),
                nn.ReLU()
            ))
            nn.init.uniform_(self.b[i].data)
        self.masker = nn.ReLU()

    def forward(self, x):
        x0=x
        for i in range(self.num_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            mask = self.masker_lst[i](self.w[i].weight)
            H = H * mask
            x = x0 * (H + self.b[i]) + x
        return x



class VIT2DEmbeddingModel(nn.Module):
    def __init__(self, 
                 vit_input_dim, 
                 vit_patch_size,
                 vit_hidden_dim, 
                 vit_num_layers, 
                 vit_num_heads,
                 **kawrgs):
        super(VIT2DEmbeddingModel, self).__init__()
        assert vit_input_dim % vit_patch_size == 0, "Input dimension must be divisible by patch size"

        self.input_dim = vit_input_dim
        self.patch_size = vit_patch_size
        self.hidden_dim = vit_hidden_dim

        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=vit_hidden_dim,
            kernel_size=vit_patch_size,
            stride=vit_patch_size
        )

        self.class_token = nn.Parameter(torch.zeros(1, 1, vit_hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=vit_hidden_dim, nhead=vit_num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=vit_num_layers)

        self.output_layer = nn.Linear(vit_hidden_dim, vit_input_dim)


    def forward(self, x):
        if(len(x.shape) == 2):
            x = x.unsqueeze(0).unsqueeze(0)
        elif(len(x.shape) == 3):
            x = x.unsqueeze(1)

        x = self.patch_embedding(x)
        batch_size, hidden_dim, num_patches_h, num_patches_w = x.size()
        num_patches = num_patches_h * num_patches_w

        x = x.flatten(2)
        x = x.transpose(1, 2)

        cls_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim)).to(x.device)
        x = x + position_embedding

        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x[0, :, :]
        x = self.output_layer(x)
        x = x.squeeze(0)
        return x