import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_regularizer
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetV2, CrossNetMix
import torch.nn.functional as F

class GlobalPooling(nn.Module):
    def __init__(self, pooling_type='mean', pooling_dim=2):
        super(GlobalPooling, self).__init__()
        self.pooling_type = pooling_type
        self.pooling_dim = pooling_dim
        
    def forward(self, x):
        if self.pooling_type == 'mean':
            return torch.mean(x, dim=self.pooling_dim )
        elif self.pooling_type == 'sum':
            return torch.sum(x, dim=self.pooling_dim )
        elif self.pooling_type == 'max':
            return torch.max(x, dim=self.pooling_dim )[0]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

class DiffPool(nn.Module):
    def __init__(self, in_channels, num_clusters):
        super(DiffPool, self).__init__()
        self.embed = SAGEConv(in_channels, num_clusters)
        
    def forward(self, x, adj_norm):
        """
        x: (batch_size, num_nodes, in_channels)
        adj_norm: (num_nodes, num_nodes) or (batch_size, num_nodes, num_nodes)
        """
        # Ensure adj_norm is 3D
        if adj_norm.dim() == 2:
            adj_norm = adj_norm.unsqueeze(0).expand(x.size(0), -1, -1)
            
        # Get assignment matrix
        s = F.softmax(self.embed(x, adj_norm), dim=-1)  # (batch_size, num_nodes, num_clusters)
        
        # Pooled features
        x_pool = torch.bmm(s.transpose(1, 2), x)  # (batch_size, num_clusters, in_channels)
        
        # Pooled adjacency matrix
        adj_pool = torch.bmm(torch.bmm(s.transpose(1, 2), adj_norm), s)
        
        # Calculate auxiliary losses - 스칼라로 변환
        adj_loss = torch.norm(adj_norm - torch.bmm(s, s.transpose(1, 2)), p=2, dim=(1,2)).mean()
        ent_loss = (-s * torch.log(s + 1e-7)).sum(dim=-1).mean()
        
        return x_pool, adj_pool, adj_loss, ent_loss


class AttentionPooling(nn.Module):
    def __init__(self, in_channels, hidden_channels, pooling_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )
        self.pooling_dim=pooling_dim
        
    def forward(self, x):
        weights = self.attention(x)
        weights = F.softmax(weights, dim=self.pooling_dim)
        weighted_x = torch.sum(weights * x, dim=self.pooling_dim)
        return weighted_x


class SAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, nomalize_adj=True):
        super(SAGEConv, self).__init__()
        self.linear_self = nn.Linear(in_channels, out_channels)
        self.linear_neigh = nn.Linear(in_channels, out_channels)
        self.nomalize_adj = nomalize_adj

        # Xavier Initialization
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x, adj):
        """
        x: 노드 특성 행렬 (num_nodes, in_channels)
        adj: 인접 행렬 (num_nodes, num_nodes)
        """
        # 자기 자신의 변환
        out_self = self.linear_self(x)

        # 이웃 노드들의 평균 계산
        # adj 행렬을 정규화 (각 행의 합이 1이 되도록)
        if self.nomalize_adj:
            deg = adj.sum(dim=1, keepdim=True)
            adj_norm = adj / (deg + 1e-6)  # 0으로 나누는 것을 방지
        else:
            adj_norm = adj

        # 이웃 노드들의 특성을 집계
        neigh_features = adj_norm @ x
        out_neigh = self.linear_neigh(neigh_features)

        # print(out_neigh.shape, out_self.shape)

        # 자기 자신과 이웃의 특성을 결합
        out = out_self + out_neigh
        return out
    

class SAGEConv2(nn.Module):
    def __init__(self, in_channels, out_channels, nomalize_adj=True):
        super(SAGEConv2, self).__init__()
        # self.linear_self = nn.Linear(in_channels, out_channels)
        # self.linear_neigh = nn.Linear(in_channels, out_channels)
        self.linear = nn.Linear(in_channels*2, out_channels)
        self.nomalize_adj = nomalize_adj

        nn.init.xavier_uniform_(self.linear.weight.data)
        # nn.init.xavier_uniform_(self.linear_neigh.weight.data)

    def forward(self, x, adj):
        """
        x: 노드 특성 행렬 (num_nodes, in_channels)
        adj: 인접 행렬 (num_nodes, num_nodes)
        """
        # 자기 자신의 변환
        # out_self = self.linear_self(x)

        # 이웃 노드들의 평균 계산
        # adj 행렬을 정규화 (각 행의 합이 1이 되도록)
        if self.nomalize_adj:
            deg = adj.sum(dim=1, keepdim=True)
            adj_norm = adj / (deg + 1e-6)  # 0으로 나누는 것을 방지
        else:
            adj_norm = adj

        # 이웃 노드들의 특성을 집계
        neigh_features = adj_norm @ x

        h_concat = torch.cat([x, neigh_features], dim=-1)
        h_out = self.linear(h_concat)
        h_out = F.relu(h_out)

        return h_out


class SAGEConv3(nn.Module):
    def __init__(self, in_channels, out_channels, nomalize_adj=True):
        super(SAGEConv3, self).__init__()
        self.linear_self = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, out_channels)
        )
        self.linear_neigh = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, out_channels)
        )
        self.nomalize_adj = nomalize_adj

        # Xavier Initialization
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, adj):
        """
        x: 노드 특성 행렬 (num_nodes, in_channels)
        adj: 인접 행렬 (num_nodes, num_nodes)
        """
        # 자기 자신의 변환
        out_self = self.linear_self(x)

        # 이웃 노드들의 평균 계산
        # adj 행렬을 정규화 (각 행의 합이 1이 되도록)
        if self.nomalize_adj:
            deg = adj.sum(dim=1, keepdim=True)
            adj_norm = adj / (deg + 1e-6)  # 0으로 나누는 것을 방지
        else:
            adj_norm = adj

        # 이웃 노드들의 특성을 집계
        neigh_features = adj_norm @ x
        out_neigh = self.linear_neigh(neigh_features)

        # print(out_neigh.shape, out_self.shape)
        # 자기 자신과 이웃의 특성을 결합
        out = out_self + out_neigh
        return out


class SAGEConv4(nn.Module):
    def __init__(self, in_channels, out_channels, num_fields, nomalize_adj=True):
        super(SAGEConv4, self).__init__()
        self.linear_self = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, out_channels)
        )
        self.linear_neigh = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, out_channels)
        )

        self.nomalize_adj = nomalize_adj
        self.positional_encoding = nn.Parameter(torch.ones((num_fields, out_channels)), requires_grad=True)

        # Xavier Initialization
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.positional_encoding)

    def forward(self, x, adj):
        """
        x: 노드 특성 행렬 (num_nodes, in_channels)
        adj: 인접 행렬 (num_nodes, num_nodes)
        """
        x = x + self.positional_encoding
        # 자기 자신의 변환
        out_self = self.linear_self(x)

        # 이웃 노드들의 평균 계산
        # adj 행렬을 정규화 (각 행의 합이 1이 되도록)
        if self.nomalize_adj:
            deg = adj.sum(dim=1, keepdim=True)
            adj_norm = adj / (deg + 1e-6)  # 0으로 나누는 것을 방지
        else:
            adj_norm = adj

        # 이웃 노드들의 특성을 집계
        neigh_features = adj_norm @ x
        out_neigh = self.linear_neigh(neigh_features)

        # print(out_neigh.shape, out_self.shape)
        # 자기 자신과 이웃의 특성을 결합
        out = out_self + out_neigh
        return out