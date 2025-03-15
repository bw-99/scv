import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter as Param
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block

class CrossNetwork(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 layer_norm=True,
                 batch_norm=True,
                 num_tower=2,
                 pooling_method="attn",
                 net_dropout=0.1,
                 num_mask=3):
        super(CrossNetwork, self).__init__()
        self.num_tower = num_tower
        self.layer_norm_flag = layer_norm
        self.batch_norm_flag = batch_norm
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.pooling_method=pooling_method
        self.num_mask = num_mask

        self.gnn_transform_weight = Param(torch.Tensor(1, num_tower, 1, 2*embedding_dim, embedding_dim))
        self.gnn_transform_bias = Param(torch.Tensor(num_tower, 1, num_fields, embedding_dim))
        nn.init.xavier_uniform_(self.gnn_transform_weight)
        nn.init.xavier_uniform_(self.gnn_transform_bias)

        self.masker = nn.Parameter(torch.zeros((self.num_tower, self.num_mask, num_fields, num_fields)))
        nn.init.xavier_uniform_(self.masker)
        self.diag_adj = nn.Parameter(torch.eye(self.num_fields, self.num_fields).unsqueeze(dim=0), requires_grad=False)
        
        self.layer_norm = nn.LayerNorm(num_fields)

        
        self.gating_network = nn.Sequential(
            nn.Linear(embedding_dim * num_fields, 1),
            nn.Softmax(dim=1)
        )


    def forward(self, _feat):
        """
        x: (batch_size, num_fields, embedding_dim)
        Output: (batch_size, num_tower * embedding_dim)
        """
        # Expand x for multiple towers => (B, T, N, D)
        x = _feat.unsqueeze(1).repeat(1, self.num_tower, 1, 1)

        # shape => (num_tower, num_hops, num_mask, N, N)
        adj_matrix_hop = torch.prod(self.masker, dim=1) # (num_tower, num_fields, num_fields)
        adj_matrix_hop = F.relu(adj_matrix_hop)
        mask = (adj_matrix_hop != 0).float()
        adj_matrix_hop = self.layer_norm(adj_matrix_hop.transpose(-2, -1)).transpose(-2, -1)
        x_masked = adj_matrix_hop + (1 - mask) * -1e9 + self.diag_adj
        adj_matrix_hop = torch.nn.functional.softmax(x_masked, dim=1) * mask

        # 1) Linear transform => (B, T, N, D)
        nei_features = torch.matmul(x.transpose(-2, -1), adj_matrix_hop).transpose(-2, -1)
        nei_features = nei_features
        
        idx = 0
        transform = self.gnn_transform_weight[:, :, idx, ...]
        x = torch.cat([x, nei_features], dim=-1) # B, T, N, 2D
        
        
        x = torch.matmul(x, transform) + self.gnn_transform_bias[:, idx, ...]
        x = x.view(x.shape[0],x.shape[1],-1)

        weights = self.gating_network(x.view(x.shape[0], x.shape[1], -1)) # B, T, N*D -> B, T, 1
        
        x = torch.sum(x * weights, dim=1).view(_feat.shape[0], _feat.shape[1], _feat.shape[2])
        return x


class SCV_coe(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="SCV_coe",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0.1,
                 num_tower=2,
                 num_hops=1,
                 num_mask=3,
                 pooling_method="attn",
                 use_bilinear_fusion=False,
                 distill_loss=None,
                 parallel_dnn_hidden_units= [400,400,400],
                 layer_norm=False,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 alpha=0.9,
                 **kwargs):
        super(SCV_coe, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.num_tower = num_tower
        self.num_tower = num_tower
        self.gpu = gpu
        self.num_hops = num_hops
        self.use_bilinear_fusion = use_bilinear_fusion
        self.num_mask = num_mask
        self.distill_loss = distill_loss

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.distill_criterion = LoCaDistillationLoss(alpha)
        input_dim = feature_map.sum_emb_out_dim()

        print("num fields", feature_map.get_num_fields())
        self.num_fields = feature_map.get_num_fields()
        self.input_dim = input_dim
        print("SCV_coe input_dim", input_dim)

        self.gnn_tower = nn.ModuleList([
            CrossNetwork(
                num_fields=self.num_fields,
                embedding_dim=embedding_dim,
                net_dropout=net_dropout,
                num_tower=num_tower,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                pooling_method=pooling_method,
                num_mask=num_mask,
            ) for _ in range(num_hops)
        ] )
        
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        
        for _ in range(num_hops):
            if batch_norm:
                # self.batch_norm.append(nn.BatchNorm1d(self.num_tower * self.num_fields * embedding_dim))
                self.batch_norm.append(nn.LayerNorm(embedding_dim))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))

        self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                      output_dim=None,  # output hidden layer
                                      hidden_units=parallel_dnn_hidden_units,
                                      hidden_activations="ReLU",
                                      output_activation=None,
                                      dropout_rates=net_dropout,
                                      batch_norm=batch_norm)

        
        if self.use_bilinear_fusion:
            concat_dim = embedding_dim * self.num_fields
            print("concat_dim ", concat_dim)
            self.bias = nn.Parameter(torch.tensor(0.0))
            self.w1 = nn.Parameter(torch.randn(concat_dim, 1))  # Shape: (d1, 1)
            self.w2 = nn.Parameter(torch.randn(parallel_dnn_hidden_units[-1], 1))  # Shape: (d2, 1)
            self.W3 = nn.Parameter(torch.randn(concat_dim, parallel_dnn_hidden_units[-1]))  # Shape: (d1, d2)
            nn.init.xavier_uniform_(self.w1)
            nn.init.xavier_uniform_(self.w2)
            nn.init.xavier_uniform_(self.W3)
        else:
            concat_dim = embedding_dim + parallel_dnn_hidden_units[-1]
            print("concat_dim ", concat_dim)
            self.scorer = nn.Sequential(
                nn.Linear(concat_dim, concat_dim),
                nn.ReLU(),
                nn.Linear(concat_dim, 1)
            )

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        print("without emb dim")
        self.count_parameters(count_embedding=False)

    def forward(self, inputs):
        feature_emb = self.embedding_layer(self.get_inputs(inputs), flatten_emb=False)

        # B, N, D
        graph_embeddings = feature_emb
        for idx in range(self.num_hops):
            x = self.gnn_tower[idx](graph_embeddings)
            
            if len(self.batch_norm) > idx:
                # x_reshaped = x.view(x.size(0), -1)  # => (B, N*D)
                x = self.batch_norm[idx](x)
                # x = x_reshaped.view(x.size(0), self.num_fields, -1)
            
            if len(self.dropout) > idx:
                x = self.dropout[idx](x)
                
            # graph_embeddings = x + graph_embeddings
            graph_embeddings = x
        
        graph_embeddings = graph_embeddings.view(feature_emb.shape[0], -1)
        mlp_embeddings = self.parallel_dnn(feature_emb.view(feature_emb.shape[0], -1))

        linear_term1 = graph_embeddings @ self.w1  # (B, output_dim)
        linear_term2 = mlp_embeddings @ self.w2  # (B, output_dim)
        bilinear_term = torch.einsum('bi,ij,bj->b', graph_embeddings, self.W3, mlp_embeddings).unsqueeze(1)  # (B, 1)
        y_pred = self.bias + linear_term1 + linear_term2 + bilinear_term

        y_pred = self.output_activation(y_pred)

        bias = self.bias / 3 if self.distill_loss == "with_bilinear" else self.bias / 2
        return_dict = {"y_pred": y_pred, "y1": linear_term1 + bias, "y2": linear_term2 + bias, "y3": bilinear_term + bias}
        return return_dict
    
    def compute_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        loss += self.regularization_loss()
        if self.distill_loss:
            y1 = self.output_activation(return_dict["y1"])
            y2 = self.output_activation(return_dict["y2"])
            loss1 = self.distill_criterion(y1, return_dict["y_pred"].detach(), y_true)
            loss2 = self.distill_criterion(y2, return_dict["y_pred"].detach(), y_true)
            loss += loss1 + loss2
        if self.distill_loss == "with_bilinear":
            y3 = self.output_activation(return_dict["y3"])
            loss3 = self.distill_criterion(y3, return_dict["y_pred"].detach(), y_true)
            loss += loss3

        return loss
    
    def distillation_loss(self, student_output, teacher_output):
        bce_loss = F.binary_cross_entropy(student_output, teacher_output, reduction='mean')
        return bce_loss
    

class LoCaDistillationLoss(nn.Module):
    def __init__(self, alpha=0.9):
        super(LoCaDistillationLoss, self).__init__()
        self.alpha = alpha  # 보정 강도 설정

    def forward(self, student_output, teacher_output, true_labels):
        
        teacher_probs_corrected = torch.where(
            true_labels == 1,  # 실제 정답이 1인 경우
            self.alpha * teacher_output + (1 - self.alpha),  # 정답 클래스 확률 ↑
            self.alpha * teacher_output  # 오답 클래스 확률 ↓
        )

        # # Step 3: 보정된 확률 (sigmoid 다시 적용)
        # teacher_probs_corrected = torch.sigmoid(teacher_logits_corrected)

        # Step 4: 보정된 교사 확률을 이용한 BCE 손실 계산
        loss = F.binary_cross_entropy(student_output, teacher_probs_corrected, reduction='mean')

        return loss