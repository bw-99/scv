import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_regularizer
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
import torch.nn.functional as F
import math
from .util import *
import logging
import tempfile
from pathlib import Path
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from tqdm import tqdm
import sys
import numpy as np

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
                 num_hops=1,
                 use_same_adj=False,
                 nomalize_adj=True,
                 layer_norm=False,
                 batch_norm=False,
                 pooling_dim=2,
                 pooling_type="mean",
                 fusion_type="MLP",
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
        self.num_hops = num_hops

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
            nomalize_adj=self.nomalize_adj,
            num_hops=num_hops
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
        # self.scorer = nn.Sequential(
        #     nn.Linear(concat_dim, concat_dim),
        #     nn.ReLU(),
        #     nn.Linear(concat_dim, concat_dim),
        #     nn.ReLU(),
        #     nn.Linear(concat_dim, 1)
        # )
        self.scorer = globals()[f"Fusion{fusion_type}"](
            num_fields=num_tower+1,
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
            nomalize_adj=self.nomalize_adj,
            num_hops=num_hops,
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
    
    def eval_step(self):
        logging.info('BO Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())    
        self.checkpoint_and_earlystop(val_logs)
        self.train()
        
    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            
            # checkpoint_data = {
            #     "epoch": self._epoch_index + 1,
            # }
            # with tempfile.TemporaryDirectory() as checkpoint_dir:
            #     data_path = Path(checkpoint_dir) / "data.pkl"
            #     with open(data_path, "wb") as fp:
            #         pickle.dump(checkpoint_data, fp)

            #     checkpoint = Checkpoint.from_directory(checkpoint_dir)
            #     train.report(
            #         {"logloss": val_logs['logloss'], "AUC": val_logs['AUC']},
            #         checkpoint=checkpoint,
            #     )
            
            return val_logs

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
                 net_dropout=0.1,
                 num_hops=1):
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
        self.num_hops=num_hops

        if self.use_same_adj:
            self.masker = nn.Parameter(torch.zeros((self.num_tower, self.num_mask, num_fields, num_fields)))
        else:
            self.masker = nn.Parameter(torch.zeros((self.num_tower, self.num_hops, self.num_mask, num_fields, num_fields)))

        nn.init.xavier_uniform_(self.masker)

        # 컨볼루션 레이어 초기화
        self.conv_lst = nn.ModuleList([
            SAGEConv3(embedding_dim, embedding_dim, num_towers=self.num_tower, nomalize_adj=nomalize_adj) for _ in range(num_hops)
        ])

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

        if self.use_same_adj:
            # (num_tower, num_mask, num_fields, num_fields)
            adj_matrix = torch.prod(self.masker, dim=1) # (num_tower, num_fields, num_fields)
            adj_matrix = F.relu(adj_matrix)
            adj_matrix = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1, -1)

            for idx in range(self.num_hops):
                x = self.conv_lst[idx](x, adj_matrix)
        else:
            # (num_tower, num_hops, num_mask, num_fields, num_fields)
            for idx in range(self.num_hops):
                # (num_tower, num_mask, num_fields, num_fields)
                adj_matrix_hop = self.masker[:, idx, ...]
                # mask 차원을 곱해 adj_matrix 생성
                adj_matrix_hop = torch.prod(adj_matrix_hop, dim=1) # (num_tower, num_fields, num_fields)
                adj_matrix_hop = F.relu(adj_matrix_hop)
                # batch 차원 확장
                adj_matrix_hop = adj_matrix_hop.unsqueeze(0).expand(batch_size, -1, -1, -1)
                x = self.conv_lst[idx](x, adj_matrix_hop)


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
            deg = adj.sum(dim=-1, keepdim=True) + 1e-6  # (batch_size, num_towers, num_fields, 1)
            # deg_inv_sqrt = deg.pow(-0.5)
            # adj = adj * deg_inv_sqrt * deg_inv_sqrt.transpose(-1, -2)
            adj = adj / (deg)

        agg = torch.matmul(adj, x)  # (batch_size, num_towers, num_fields, embedding_dim)
        agg = self.agg_lin(agg)

        out = out + agg
        # out = F.relu(out)
        return out  # (batch_size, num_towers, num_fields, out_channels)

class FusionMLP(nn.Module):
    def __init__(self, embedding_dim, num_fields, **kawrgs):
        super(FusionMLP, self).__init__()
        concat_dim = (num_fields) * embedding_dim
        self.fusion_network = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, 1),
        )
    
    def forward(self, x):
        return self.fusion_network(x)
    
class FusionMaxPooling(nn.Module):
    def __init__(self, embedding_dim, num_fields, **kawrgs):
        super(FusionMaxPooling, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        concat_dim = embedding_dim
        self.scorer = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, 1),
        )
    def forward(self, x):
        x = x.view(-1, self.num_fields, self.embedding_dim)
        return self.scorer(torch.max(x, dim=1)[0])
    
class FusionMeanPooling(nn.Module):
    def __init__(self, embedding_dim, num_fields, **kawrgs):
        super(FusionMeanPooling, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        concat_dim = embedding_dim
        self.scorer = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, 1),
        )
    def forward(self, x):
        x = x.view(-1, self.num_fields, self.embedding_dim)
        return self.scorer(torch.mean(x, dim=1))
    

class FusionATTN(nn.Module):
    def __init__(self, embedding_dim, num_fields, num_heads=8, **kawrgs):
        super(FusionATTN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads."

        concat_dim = num_fields * embedding_dim

        # Query, Key, Value projection
        self.query_layer = nn.Linear(embedding_dim, embedding_dim)
        self.key_layer = nn.Linear(embedding_dim, embedding_dim)
        self.value_layer = nn.Linear(embedding_dim, embedding_dim)

        # Output projection after multi-head attention
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Fusion network to combine node outputs
        self.fusion_network = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, 1),
        )

    def forward(self, x):
        # x shape: (batch_size, num_fields, embedding_dim)
        x = x.view(-1, self.num_fields, self.embedding_dim)
        B, N, D = x.size()
        
        # Linear projections
        Q = self.query_layer(x) # (B, N, D)
        K = self.key_layer(x)   # (B, N, D)
        V = self.value_layer(x) # (B, N, D)

        # Split into multiple heads
        # Resulting shape: (B, num_heads, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # attn_weights: (B, num_heads, N, N)
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn_weights, dim=-1)

        # Apply attention to V
        # out: (B, num_heads, N, head_dim)
        out = torch.matmul(attn, V)

        # Concat heads
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        # Final projection and fusion
        out = self.out_proj(out) # (B, N, D)
        
        # Flatten before fusion network
        out = out.view(B, N * D)
        out = self.fusion_network(out)

        return out


class FusionGAS(nn.Module):
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
                 net_dropout=0.1,
                 num_hops=1, 
                 **kawrgs):
        super(FusionGAS, self).__init__()
        self.fusion_netowrk = CrossNetwork(
            num_fields,
            embedding_dim,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            use_same_adj=use_same_adj,
            num_mask=num_mask,
            pooling_type=pooling_type,
            pooling_dim=pooling_dim,
            gpu=gpu,
            nomalize_adj=nomalize_adj,
            num_tower=num_tower,
            net_dropout=net_dropout,
            num_hops=num_hops
        )
        
        concat_dim = embedding_dim * num_tower
        self.embedding_dim= embedding_dim
        self.num_tower= num_tower
        self.num_fields = num_fields
        self.scorer = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, 1),
        )
        
    def forward(self, x):
        x = x.view(-1, self.num_fields, self.embedding_dim)
        x = self.fusion_netowrk(x)
        return self.scorer(x)