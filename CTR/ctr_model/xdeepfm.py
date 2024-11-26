# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from CTR.ctr_model.layers import Embeddings, MLPBlock, CIN, GatedMultimodalUnit

class xDeepFM(nn.Module):
    def __init__(self, input_size,
                 num_fields,
                 embed_size=128,
                 hidden_size=256,
                 num_hidden_layers=6,
                 embed_dropout_rate=0.1,
                 hidden_dropout_rate=0.3,
                 hidden_act='relu',
                 out_dim=1,
                 cin_layer_units="25-25-25",
                 vec_dim=0,
                 fusion_type="logit_add"):
        super(xDeepFM, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.out_dim = out_dim
        self.vec_dim = vec_dim
        self.fusion_type = fusion_type

        self.embed = Embeddings(input_size, embed_size, embed_dropout_rate)
        input_dim = num_fields * embed_size
        cin_layer_units = [int(c) for c in cin_layer_units.split('-')]
        self.cin = CIN(num_fields, cin_layer_units)
        self.final_dim = sum(cin_layer_units)
        if self.num_hidden_layers > 0:
            self.dnn = MLPBlock(input_dim=input_dim,
                                hidden_size=hidden_size,
                                num_hidden_layers=self.num_hidden_layers,
                                hidden_dropout_rate=hidden_dropout_rate,
                                hidden_act=hidden_act)
            self.final_dim += hidden_size

        if self.vec_dim:
            self.vec_trans = nn.Sequential(
                nn.Linear(self.vec_dim, self.final_dim),
                nn.ReLU(),
                nn.Linear(self.final_dim, self.final_dim)
            )
            if self.fusion_type == "gate_fusion":
                self.fusion = GatedMultimodalUnit(self.final_dim, self.final_dim, self.final_dim)
            elif self.fusion_type == "logit_add":
                self.vec_out = nn.Linear(self.final_dim, self.out_dim)
            elif self.fusion_type == "gate_logit_add":
                self.fusion = GatedMultimodalUnit(self.final_dim, self.final_dim, self.final_dim)
                self.vec_out = nn.Sequential(
                    nn.Linear(self.final_dim, self.final_dim),
                    nn.ReLU(),
                    nn.Linear(self.final_dim, self.out_dim)
                )
            elif self.fusion_type == "only_vec":
                self.vec_out = nn.Linear(self.final_dim, self.out_dim)

        self.fc_out = nn.Linear(self.final_dim, self.out_dim)

    def forward(self, input_ids):
        if self.vec_dim:
            input_ids, input_vec = input_ids

        feat_embed = self.embed(input_ids)
        final_vec = self.cin(feat_embed)
        if self.num_hidden_layers > 0:
            dnn_vec = self.dnn(feat_embed.flatten(start_dim=1))
            final_vec = torch.cat([final_vec, dnn_vec], dim=1)
        
        if self.vec_dim:
            vec_output =self.vec_trans(input_vec)

            if self.fusion_type == "gate_fusion":
                fusion_out = self.fusion(dnn_vec, vec_output)
                logits = self.fc_out(fusion_out)

            elif self.fusion_type == "logit_add":
                vec_logits = self.vec_out(vec_output)
                logits = self.fc_out(dnn_vec) + vec_logits

            elif self.fusion_type == "gate_logit_add":
                fusion_out = self.fusion(dnn_vec, vec_output)
                vec_logits = self.vec_out(fusion_out)
                logits = self.fc_out(dnn_vec) + vec_logits
            elif self.fusion_type == "only_vec":
                vec_logits = self.vec_out(vec_output)
                logits = vec_logits

            logits = logits.squeeze()
            logits = torch.sigmoid(logits)
            return logits


        logits = self.fc_out(final_vec)
        logits = logits.squeeze()
        logits = torch.sigmoid(logits)
        return logits