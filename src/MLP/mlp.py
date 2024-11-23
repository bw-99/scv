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

class MLP(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="MLP",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0.1,
                 layer_norm=True,
                 batch_norm=True,
                 embedding_regularizer=None,
                 parallel_dnn_hidden_units= [400,400,400],
                 net_regularizer=None,
                 **kwargs):
        super(MLP, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.gpu = gpu

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()

        print("num feileds", feature_map.get_num_fields())
        self.num_fields = feature_map.get_num_fields()
        self.input_dim = input_dim
        print("MLPMLPMLPMLPMLP input_dim", input_dim)


        self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                          output_dim=1, # output hidden layer
                                          hidden_units=parallel_dnn_hidden_units,
                                          hidden_activations="ReLU",
                                          output_activation=None, 
                                          dropout_rates=net_dropout, 
                                          batch_norm=batch_norm)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        self.tmp_val_lst = []
        self.val_lst = []

        print("without emb dim")
        self.count_parameters(count_embedding=False)

        

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        flattened_emb = self.embedding_layer(X, flatten_emb=True)
        dnn_emb = self.parallel_dnn(flattened_emb)
        y_pred = self.output_activation(dnn_emb)

        return_dict = {"y_pred": y_pred}
        return return_dict
