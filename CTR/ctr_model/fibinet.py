# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from CTR.ctr_model.layers import Embeddings, MLPBlock, SqueezeExtractionLayer, BilinearInteractionLayer, GatedMultimodalUnit


class FiBiNet(nn.Module):
    def __init__(self,
                 input_size,
                 num_fields,
                 embed_size=128,
                 hidden_size=256,
                 num_hidden_layers=6,
                 embed_dropout_rate=0.1,
                 hidden_dropout_rate=0.3,
                 hidden_act='relu',
                 out_dim=1,
                 reduction_ratio=0.1,
                 bilinear_type="field_all",
                 vec_dim=0,
                 fusion_type="logit_add"
                 ):
        super(FiBiNet, self).__init__()
        self.out_dim = out_dim
        self.vec_dim = vec_dim
        self.fusion_type = fusion_type
        self.num_hidden_layers = num_hidden_layers

        self.embed = Embeddings(input_size, embed_size, embed_dropout_rate)
        self.senet_layer = SqueezeExtractionLayer(num_fields, reduction_ratio)
        self.bilinear_layer = BilinearInteractionLayer(bilinear_type, num_fields, embed_size)
        self.final_dim = num_fields * (num_fields - 1) * embed_size
        if num_hidden_layers > 0:
            self.dnn = MLPBlock(input_dim=self.final_dim,
                                hidden_size=hidden_size,
                                num_hidden_layers=num_hidden_layers,
                                hidden_dropout_rate=hidden_dropout_rate,
                                hidden_act=hidden_act)
            self.final_dim = hidden_size

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
        senet_embed = self.senet_layer(feat_embed)
        bilinear_p = self.bilinear_layer(feat_embed)
        bilinear_q = self.bilinear_layer(senet_embed)
        final_output = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        if self.num_hidden_layers > 0:
            final_output = self.dnn(final_output)

        
        if self.vec_dim:
            vec_output =self.vec_trans(input_vec)

            if self.fusion_type == "gate_fusion":
                fusion_out = self.fusion(final_output, vec_output)
                logits = self.fc_out(fusion_out)

            elif self.fusion_type == "logit_add":
                vec_logits = self.vec_out(vec_output)
                logits = self.fc_out(final_output) + vec_logits

            elif self.fusion_type == "gate_logit_add":
                fusion_out = self.fusion(final_output, vec_output)
                vec_logits = self.vec_out(fusion_out)
                logits = self.fc_out(final_output) + vec_logits
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
