import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter as Param
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block

class GasStream(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 layer_norm=True,
                 batch_norm=True,
                 num_tower=2,
                 net_dropout=0.1,
                 num_mask=3,
                 num_hops=1,
                ):
        super(GasStream, self).__init__()
        self.num_tower = num_tower
        self.layer_norm_flag = layer_norm
        self.batch_norm_flag = batch_norm
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.num_hops = num_hops
        self.num_mask = num_mask

        self.gnn_transform_weight = Param(torch.Tensor(1, num_tower, num_hops, 2*embedding_dim, embedding_dim))
        self.gnn_transform_bias = Param(torch.Tensor(num_tower, num_hops, num_fields, embedding_dim))
        
        self.masker = nn.Parameter(torch.zeros((self.num_tower, self.num_mask, num_fields, num_fields)))
        self.diag_adj = nn.Parameter(torch.eye(self.num_fields, self.num_fields).unsqueeze(dim=0), requires_grad=False)
        
        self.layer_norm = nn.LayerNorm(num_fields)
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        for _ in range(num_hops):
            if batch_norm:
                dim = self.num_tower * self.num_fields * 2 * self.embedding_dim
                self.batch_norm.append(nn.BatchNorm1d(dim))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
        
        self.feature_moe_gate = nn.Sequential(
            nn.Linear(embedding_dim * num_fields, num_fields),
            nn.Softmax(dim=2)
        )
        
        self.tower_moe_gate = nn.Sequential(
            nn.Linear(embedding_dim * num_fields, num_tower),
            nn.Softmax(dim=1)
        )

        nn.init.xavier_uniform_(self.gnn_transform_weight)
        nn.init.xavier_uniform_(self.gnn_transform_bias)
        nn.init.xavier_uniform_(self.masker)

    def forward(self, _feat):
        x = _feat.unsqueeze(1).repeat(1, self.num_tower, 1, 1)
        
        adj_matrix_hop = torch.prod(self.masker, dim=1)
        adj_matrix_hop = F.relu(adj_matrix_hop)
        mask = (adj_matrix_hop != 0).float()
        adj_matrix_hop = self.layer_norm(adj_matrix_hop.transpose(-2, -1)).transpose(-2, -1)
        x_masked = adj_matrix_hop + (1 - mask) * -1e9 + self.diag_adj
        adj_matrix_hop = torch.nn.functional.softmax(x_masked, dim=1) * mask

        for idx in range(self.num_hops):
            nei_features = torch.matmul(x.transpose(-2, -1), adj_matrix_hop).transpose(-2, -1)
            nei_features = nei_features
            transform = self.gnn_transform_weight[:, :, idx, ...]
            x = torch.cat([x, nei_features], dim=-1)
            
            if len(self.batch_norm) > idx:
                x_reshaped = x.view(x.size(0), -1)
                x_reshaped = self.batch_norm[idx](x_reshaped)
                x = x_reshaped.view(x.size(0), self.num_tower, self.num_fields, -1)
                
            x = torch.matmul(x, transform) + self.gnn_transform_bias[:, idx, ...]

            if len(self.dropout) > idx:
                x = self.dropout[idx](x)

        feature_weight = self.feature_moe_gate(x.view(x.shape[0], x.shape[1], -1))
        x = torch.sum(x * feature_weight.unsqueeze(dim=-1), dim=2)

        tower_weight = self.tower_moe_gate(_feat.view(_feat.shape[0], -1)).unsqueeze(dim=-1)
        x = torch.sum(x * tower_weight, dim=1)
        return x


class SCV(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="SCV",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0.1,
                 scv_dropout=None,
                 num_tower=2,
                 num_hops=1,
                 num_mask=3,
                 parallel_dnn_hidden_units=[400,400,400],
                 layer_norm=False,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 gamma=0.9,
                 **kwargs):
        super(SCV, self).__init__(feature_map,
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
        self.num_mask = num_mask

        input_dim = feature_map.sum_emb_out_dim()
        self.num_fields = feature_map.get_num_fields()
        self.input_dim = input_dim

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.distill_criterion = LBD(gamma)

        self.gas_stream = GasStream(
            num_fields=self.num_fields,
            embedding_dim=embedding_dim,
            net_dropout=net_dropout if scv_dropout == None else scv_dropout,
            num_tower=num_tower,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            num_hops=num_hops,
            num_mask=num_mask,
        )
        
        self.mlp_stream = MLP_Block(input_dim=input_dim,
                                    output_dim=None,
                                    hidden_units=parallel_dnn_hidden_units,
                                    hidden_activations="ReLU",
                                    output_activation=None,
                                    dropout_rates=net_dropout,
                                    batch_norm=batch_norm)
                
        self.W_EH = nn.Parameter(torch.randn(embedding_dim, parallel_dnn_hidden_units[-1]))
        self.W_E = nn.Parameter(torch.randn(embedding_dim, 1))
        self.W_H = nn.Parameter(torch.randn(parallel_dnn_hidden_units[-1], 1))
        self.bias = nn.Parameter(torch.tensor(0.0))
        nn.init.xavier_uniform_(self.W_EH)
        nn.init.xavier_uniform_(self.W_E)
        nn.init.xavier_uniform_(self.W_H)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=False)

        gas_embeddings = self.gas_stream(feature_emb).view(feature_emb.shape[0], -1)
        mlp_embeddings = self.mlp_stream(feature_emb.view(feature_emb.shape[0], -1))

        logit_E = gas_embeddings @ self.W_E
        logit_H = mlp_embeddings @ self.W_H
        logit_EH = torch.einsum('bi,ij,bj->b', gas_embeddings, self.W_EH, mlp_embeddings).unsqueeze(1)
        y_pred = self.output_activation(self.bias + logit_E + logit_H + logit_EH)

        return {"y_pred": y_pred, "logit_E": logit_E, "logit_H": logit_H, "logit_EH": logit_EH}
    
    def compute_loss(self, return_dict, y_true):
        origin_loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        loss = origin_loss
        loss += self.regularization_loss()
        
        term_lst = [return_dict["logit_E"], return_dict["logit_H"]]
        term_loss = torch.tensor([self.loss_fn(self.output_activation(item), y_true, reduction='mean') for item in term_lst], device=origin_loss.device)
        term_loss = F.softmax(term_loss)
        
        y_E = self.output_activation(return_dict["logit_E"])
        y_H = self.output_activation(return_dict["logit_H"])
        loss_E = self.distill_criterion(y_E, return_dict["y_pred"].detach(), y_true) * term_loss[0]
        loss_H = self.distill_criterion(y_H, return_dict["y_pred"].detach(), y_true) * term_loss[1]
        loss += loss_E+loss_H

        return loss
    

class LBD(nn.Module):
    def __init__(self, gamma=0.9):
        super(LBD, self).__init__()
        self.gamma = gamma

    def forward(self, student_output, teacher_output, true_labels):
        teacher_probs_corrected = torch.where(
            true_labels == 1,
            self.gamma * teacher_output + (1 - self.gamma),
            self.gamma * teacher_output
        )
        loss = F.binary_cross_entropy(student_output, teacher_probs_corrected, reduction='mean')

        return loss