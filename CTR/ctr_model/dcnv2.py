# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from CTR.ctr_model.layers import Embeddings, MLPBlock, GatedMultimodalUnit, CrossNetV2
import math

class DCNV2(nn.Module):
    def __init__(self, num_fields, embed_size, input_size, embed_dropout_rate, num_cross_layers, num_hidden_layers,
                 hidden_size, hidden_dropout_rate, hidden_act, output_dim, vec_dim=0, fusion_type="logit_add"):
        super(DCNV2, self).__init__()

        self.vec_dim = vec_dim
        self.fusion_type = fusion_type
        self.embed = Embeddings(input_size, embed_size, embed_dropout_rate)
        input_dim = num_fields * embed_size
        self.cross_net = CrossNetV2(input_dim, num_cross_layers)
        self.final_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        if num_hidden_layers > 0:
            self.parallel_dnn = MLPBlock(input_dim=input_dim,
                                         hidden_size=hidden_size,
                                         num_hidden_layers=num_hidden_layers,
                                         hidden_dropout_rate=hidden_dropout_rate,
                                         hidden_act=hidden_act)
            self.final_dim += hidden_size

        if self.vec_dim:
            self.vec_trans = nn.Sequential(
                nn.Linear(vec_dim, self.final_dim),
                nn.ReLU()
            )

            if self.fusion_type == "gate_fusion":
                self.gate_fusion = GatedMultimodalUnit(self.final_dim, self.final_dim, self.final_dim)
            elif self.fusion_type == "logit_add":
                self.vec_out = nn.Linear(self.final_dim, self.output_dim)
            elif self.fusion_type == "gate_logit_add":
                self.gate_fusion = GatedMultimodalUnit(self.final_dim, self.final_dim, self.final_dim)
                self.fusion_pro = nn.Sequential(
                    nn.Linear(self.final_dim, self.final_dim),
                    nn.ReLU()
                )
                self.vec_out = nn.Linear(self.final_dim, self.output_dim)
            elif self.fusion_type == "only_vec":
                self.vec_out = nn.Linear(self.final_dim, self.output_dim)

                                     

        self.fc_out = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, input_ids):
        if self.vec_dim:
            input_ids, input_vec = input_ids

        feat_embed = self.embed(input_ids).flatten(start_dim=1)
        final_output = self.cross_net(feat_embed)

        if self.num_hidden_layers > 0:
            dnn_output = self.parallel_dnn(feat_embed)
            final_output = torch.cat([final_output, dnn_output], dim=-1)

        if self.vec_dim:
            vec_output = self.vec_trans(input_vec)

            if self.fusion_type == "logit_add":
                vec_logits = self.vec_out(vec_output)
                logits = vec_logits + self.fc_out(final_output)

            elif self.fusion_type == "gate_fusion":
                gate_out = self.gate_fusion(final_output, vec_output)
                logits = self.fc_out(gate_out)

            elif self.fusion_type == "gate_logit_add":
                gate_out = self.gate_fusion(final_output, vec_output)
                gate_out = self.fusion_pro(gate_out)
                gate_logits = self.vec_out(gate_out)
                logits = gate_logits + self.fc_out(final_output)
            
            elif self.fusion_type == "only_vec":
                vec_logits = self.vec_out(vec_output)
                logits = vec_logits


            logits = logits.squeeze()
            logits = torch.sigmoid(logits)
            return logits
            
        logits = self.fc_out(final_output)
        logits = logits.squeeze()
        logits = torch.sigmoid(logits)


        return logits
