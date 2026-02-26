import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.modules import AttentionMap, TextureEnhance, AttentionPooling

"""
Exctration des features des layers d'interet
layer shallow : layer 2
layer deep : layer 5

Les auteurs ont reconstruit le code du modèle efficient_b4
On choisit efficent_b0 par rappor à nos ressources limitées
On choisit de prendre le modèle déjà prêt et complet de la librairie timm pour une exactitude 100% du modèle
"""


class MultiAttentionNet(nn.Module):

    def __init__(self, model_name='tf_efficientnet_b0_ns', num_classes=2, M=4, dropout_rate=0.5):

        super(MultiAttentionNet, self).__init__()
        self.M = M

        # Backbone
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 380, 380)
            layers = self.backbone(dummy)
            self.feat_dim_texture = layers[1].shape[1] 
            self.feat_dim_semantic = layers[4].shape[1]

        # Modules MAT
        self.attentions = AttentionMap(self.feat_dim_semantic, self.M)
        self.texture_enhance = TextureEnhance(self.feat_dim_texture, self.M)
        self.atp = AttentionPooling()

        # Classifiers
        mid_dims = 256
        self.projection_local = nn.Sequential(
            nn.Linear(self.M * self.feat_dim_texture, mid_dims),
            nn.Hardswish(),
            nn.Linear(mid_dims, mid_dims)
        )

        self.project_final = nn.Linear(self.feat_dim_semantic, mid_dims)
        self.ensemble_classifier = nn.Sequential(
            nn.Linear(mid_dims * 2, mid_dims),
            nn.Hardswish(),
            nn.Linear(mid_dims, num_classes)
        )
        self.dropout = nn.Dropout(dropout_rate)
        

    def forward(self, x):
        features = self.backbone(x)
        feat_texture = features[1]   
        feat_semantic = features[4]  
        
        attention_maps = self.attentions(feat_semantic)
        texture_maps, texture_raw = self.texture_enhance(feat_texture, attention_maps)

        # Branche Locale
        texture_matrix = self.atp(texture_maps, attention_maps)
        B = x.size(0)
        vec_texture = self.projection_local(texture_matrix.view(B, -1))
        vec_texture = self.dropout(vec_texture)

        # Branche Globale
        att_sum = attention_maps.sum(dim=1, keepdim=True)
        vec_semantic = self.project_final(self.atp(features[4], att_sum, norm=1).squeeze(1))

        logits = self.ensemble_classifier(torch.cat((vec_texture, vec_semantic), dim=1))

        return {
            "logits": logits,
            "attention_maps": attention_maps,
            "texture_raw": texture_raw
        }