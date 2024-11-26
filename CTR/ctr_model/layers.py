# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations, product
import math


class LEU(nn.Module):
    __constants__ = ['alpha', 'inplace']
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1.):
        super(LEU, self).__init__()
        self.alpha = alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # TODO more efficient
        output = torch.empty_like(input)
        output[input > 0] = self.alpha * torch.log(input[input > 0] + 1)
        output[input <= 0] = self.alpha * (torch.exp(input[input <= 0]) - 1)
        return output

class NoneAct(nn.Module):
    def forward(self, x):
        return x


class GELU(nn.Module):
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU_new(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def get_act(act_func):
    if isinstance(act_func, str):
        if act_func.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif act_func.lower() == 'tanh':
            return nn.Tanh()
        elif act_func.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif act_func.lower() == 'none':
            return NoneAct()
        elif act_func.lower() == 'elu':
            return nn.ELU()
        elif act_func.lower() == 'leu':
            return LEU()
        elif act_func.lower() == 'gelu':
            return GELU()
        elif act_func.lower() == 'gelu_new':
            return GELU_new()
        elif act_func.lower() == 'swish':
            return Swish()
        elif act_func.lower() == 'mish':
            return Mish()
        else:
            raise NotImplementedError
    else:
        return act_func


class Embeddings(nn.Module):
    def __init__(self, input_size, embed_size, embed_dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim=embed_size)
        self.dropout = nn.Dropout(embed_dropout_rate)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        return embeddings


class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_cross_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_cross_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim) for _ in range(num_cross_layers))

    def forward(self, X0):
        Xi = X0
        for i in range(self.num_layers):
            Xi = Xi + X0 * self.cross_layers[i](Xi)
        return Xi


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_hidden_layers=3,
                 hidden_act='relu', hidden_dropout_rate=0.5, batch_norm=False):
        super(MLPBlock, self).__init__()
        dense_layers = []
        for i in range(num_hidden_layers):
            dense_layers.append(nn.Linear(input_dim, hidden_size))
            if batch_norm:
                pass
            dense_layers.append(get_act(hidden_act))
            dense_layers.append(nn.Dropout(p=hidden_dropout_rate))
            input_dim = hidden_size
        self.dnn = nn.Sequential(*dense_layers)

    def forward(self, inputs):
        return self.dnn(inputs)

class GatedMultimodalUnit(nn.Module):
    def __init__(self, input_dim_id, input_dim_vec, output_dim):
        super(GatedMultimodalUnit, self).__init__()
        self.W_id = nn.Linear(input_dim_id, output_dim)
        self.W_vec = nn.Linear(input_dim_vec, output_dim)
        self.W_gate = nn.Linear(input_dim_id + input_dim_vec, output_dim)

    def forward(self, x_id, x_vec):

        h_id = torch.tanh(self.W_id(x_id))
        h_vec = torch.tanh(self.W_vec(x_vec))

        x_cat = torch.cat([x_id, x_vec], dim=-1)
        gate_score = torch.sigmoid(self.W_gate(x_cat))

        h = gate_score * h_vec + (1 - gate_score) * h_id

        return h

class LR(nn.Module):
    def __init__(self, input_size):
        super(LR, self).__init__()
        self.embed_w = nn.Embedding(input_size, embedding_dim=1)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input_ids):
        wx = self.embed_w(input_ids)
        logits = wx.sum(dim=1) + self.bias
        return logits


class InnerProductLayer(nn.Module):
    def __init__(self, num_fields=None, output='product_sum'):
        super(InnerProductLayer, self).__init__()
        self.output_type = output
        if output not in ['product_sum', 'bi_interaction', 'inner_product', 'elementwise_product']:
            raise ValueError(f'InnerProductLayer output={output} is not supported')
        if num_fields is None:
            if output in ['inner_product', 'elementwise_product']:
                raise ValueError(f'num_fields is required when InnerProductLayer output={output}')
        else:
            p, q = zip(*list(combinations(range(num_fields), 2)))
            self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)
            self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)
            self.upper_triangle_mask = nn.Parameter(
                torch.triu(torch.ones(num_fields, num_fields), 1).type(torch.bool),
                requires_grad=False)

    def forward(self, feat_embed):
        if self.output_type in ['product_sum', 'bi_interaction']:
            sum_of_square = torch.sum(feat_embed, dim=1) ** 2  # sum then square
            square_of_sum = torch.sum(feat_embed ** 2, dim=1)  # square then sum
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self.output_type == 'bi_interaction':
                return bi_interaction
            else:
                return bi_interaction.sum(dim=-1, keepdim=True)
        elif self.output_type == 'inner_product':
            inner_product_matrix = torch.bmm(feat_embed, feat_embed.transpose(1, 2))
            flat_upper_triangle = torch.masked_select(inner_product_matrix, self.upper_triangle_mask)
            return flat_upper_triangle.view(-1, self.interaction_units)
        else:
            raise NotImplementedError

class SqueezeExtractionLayer(nn.Module):
    def __init__(self, num_fields, reduction_ratio):
        super(SqueezeExtractionLayer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_size, num_fields, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, feature_emb):
        Z = torch.mean(feature_emb, dim=-1, out=None)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V


class BilinearInteractionLayer(nn.Module):
    def __init__(self, bilinear_type, num_fields, embed_size):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(embed_size, embed_size, bias=False)
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False)
                                                 for i in range(num_fields)])
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False)
                                                 for i, j in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i) * v_j
                             for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]
                             for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1]
                             for i, v in enumerate(combinations(feature_emb_list, 2))]
        return torch.cat(bilinear_list, dim=1)


class CIN(nn.Module):
    def __init__(self, num_fields, cin_layer_units):
        super(CIN, self).__init__()
        self.cin_layer_units = cin_layer_units
        self.cin_layer = nn.ModuleDict()
        for i, unit in enumerate(self.cin_layer_units):
            in_channels = num_fields * self.cin_layer_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer["layer_" + str(i + 1)] = nn.Conv1d(in_channels,
                                                              out_channels,  # how many filters
                                                              kernel_size=1)  # kernel output shape

    def forward(self, X_0):
        pooling_outputs = []
        batch_size = X_0.shape[0]
        embedding_dim = X_0.shape[-1]
        X_i = X_0
        for i in range(len(self.cin_layer_units)):
            hadamard_tensor = torch.einsum("bhd,bmd->bhmd", X_0, X_i)
            hadamard_tensor = hadamard_tensor.view(batch_size, -1, embedding_dim)
            X_i = self.cin_layer["layer_" + str(i + 1)](hadamard_tensor) \
                .view(batch_size, -1, embedding_dim)
            pooling_outputs.append(X_i.sum(dim=-1))
        concate_vec = torch.cat(pooling_outputs, dim=-1)

        return concate_vec
