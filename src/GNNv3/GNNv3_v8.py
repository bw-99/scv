import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_regularizer
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
import torch.nn.functional as F
from .util import *

class GNNv3_v8(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="GNNv3_v8",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0.1,
                 num_tower=2,
                 num_mask=2,
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
        super(GNNv3_v8, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.num_tower = num_tower
        self.pooling_dim = pooling_dim
        self.num_tower = num_tower
        self.nomalize_adj = nomalize_adj
        self.num_mask = num_mask
        self.use_same_adj = use_same_adj
        self.gpu = gpu

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()

        print("num fields", feature_map.get_num_fields())
        self.num_fields = feature_map.get_num_fields()
        self.input_dim = input_dim
        print("GNNv3_v8 input_dim", input_dim)

        self.gnn_tower = CrossNetwork(
            num_fields=self.num_fields,
            embedding_dim=embedding_dim,
            net_dropout=net_dropout,
            num_tower=num_tower,
            num_mask=self.num_mask,
            layer_norm=layer_norm,
            pooling_type=pooling_type,
            use_same_adj=self.use_same_adj,
            pooling_dim=pooling_dim,
            batch_norm=batch_norm,
            gpu=self.gpu,
            nomalize_adj=self.nomalize_adj
        )

        final_dim = embedding_dim

        self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                      output_dim=final_dim,  # output hidden layer
                                      hidden_units=parallel_dnn_hidden_units,
                                      hidden_activations="ReLU",
                                      output_activation=None,
                                      dropout_rates=net_dropout,
                                      batch_norm=batch_norm)

        concat_dim = (self.num_tower + 1) * final_dim
        self.scorer = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, 1)
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

        gnn_emb = self.gnn_tower(feature_emb)
        dnn_emb = self.parallel_dnn(flattened_emb)

        final_emb = torch.cat([gnn_emb, dnn_emb], dim=-1)
        y_pred = self.scorer(final_emb)

        y_pred = self.output_activation(y_pred)

        return_dict = {"y_pred": y_pred}
        return return_dict

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
                 num_tower=2,
                 net_dropout=0.1):
        super(CrossNetwork, self).__init__()
        self.num_tower = num_tower
        self.nomalize_adj = nomalize_adj
        self.use_same_adj = use_same_adj
        self.num_mask = num_mask
        self.layer_norm_flag = layer_norm
        self.batch_norm_flag = batch_norm
        self.dropout = None

        self.embedding_dim = embedding_dim
        self.num_fields = num_fields

        # 마스크 초기화
        if self.use_same_adj:
            self.masker = nn.Parameter(torch.zeros((self.num_mask, num_fields, num_fields)))
            nn.init.xavier_uniform_(self.masker)
        else:
            self.masker = nn.Parameter(torch.zeros((self.num_tower, self.num_mask, num_fields, num_fields)))
            nn.init.xavier_uniform_(self.masker)

        # 컨볼루션 레이어 초기화
        self.conv = SAGEConv3(embedding_dim, embedding_dim, num_towers=self.num_tower, nomalize_adj=nomalize_adj)

        # 정규화 레이어
        if self.layer_norm_flag:
            self.layer_norm = nn.LayerNorm([self.num_tower, num_fields, embedding_dim])
        if self.batch_norm_flag:
            self.batch_norm = nn.BatchNorm1d(self.num_fields * embedding_dim)
        if net_dropout > 0:
            self.dropout = nn.Dropout(net_dropout)
        else:
            self.dropout = None

        # 풀링 레이어
        self.pool = GlobalPooling(pooling_type, pooling_dim=pooling_dim)

    def forward(self, x):
        batch_size, num_fields, embedding_dim = x.size()

        # x를 num_tower 차원으로 확장
        x = x.unsqueeze(1).repeat(1, self.num_tower, 1, 1)  # (batch_size, num_tower, num_fields, embedding_dim)

        # Adjacent Matrices 생성
        if self.use_same_adj:
            adj_matrix = torch.prod(self.masker, dim=0)
            adj_matrix = F.relu(adj_matrix)
            adj_matrix = adj_matrix.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_tower, -1, -1)
        else:
            mask_stack = torch.prod(self.masker, dim=1)  # (num_tower, num_fields, num_fields)
            adj_matrices = F.relu(mask_stack)
            adj_matrix = adj_matrices.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Convolution 연산을 벡터화하여 수행
        x = self.conv(x, adj_matrix)  # (batch_size, num_tower, num_fields, embedding_dim)

        # Normalization 적용
        if self.batch_norm_flag:
            x_reshaped = x.view(batch_size * self.num_tower, -1)
            x_reshaped = self.batch_norm(x_reshaped)
            x = x_reshaped.view(batch_size, self.num_tower, self.num_fields, self.embedding_dim)
        if self.layer_norm_flag:
            x = self.layer_norm(x)

        # 필요에 따라 Dropout 적용
        if self.dropout:
            x = self.dropout(x)

        # 각 헤드별 출력을 평균 또는 다른 방식으로 결합
        x = x.mean(dim=2)  # (batch_size, num_tower, embedding_dim)
        x = x.view(batch_size, embedding_dim * self.num_tower)
        return x

class SAGEConv3(nn.Module):
    def __init__(self, in_channels, out_channels, num_towers=1, nomalize_adj=True):
        super(SAGEConv3, self).__init__()
        self.num_towers = num_towers
        self.nomalize_adj = nomalize_adj
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        # x: (batch_size, num_towers, num_fields, embedding_dim)
        # adj: (batch_size, num_towers, num_fields, num_fields)
        batch_size, num_towers, num_fields, embedding_dim = x.size()

        # Apply linear transformation
        out = self.lin(x)  # (batch_size, num_towers, num_fields, out_channels)

        # Neighbor aggregation
        if self.nomalize_adj:
            # Degree normalization
            deg = adj.sum(dim=-1, keepdim=True) + 1e-7  # (batch_size, num_towers, num_fields, 1)
            deg_inv_sqrt = deg.pow(-0.5)
            adj = adj * deg_inv_sqrt * deg_inv_sqrt.transpose(-1, -2)

        agg = torch.matmul(adj, x)  # (batch_size, num_towers, num_fields, embedding_dim)
        agg = self.agg_lin(agg)

        out += agg
        out = F.relu(out)
        return out  # (batch_size, num_towers, num_fields, out_channels)
