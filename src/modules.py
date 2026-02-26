import torch
import torch.nn as nn
import torch.nn.functional as F

"""
S'inspire pas mal du github qui présente des variations par rapport au papier (codes v2)
"""


class AttentionMap(nn.Module):
    """
    Bloc B : Génère M cartes d'attention à partir des features sémantiques (High Level)
    """
    def __init__(self, in_channels, out_channels):
        super(AttentionMap, self).__init__()
        self.num_attentions = out_channels

        # Convolution pour extraire l'information pertinente
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Convolution 1x1 pour générer les M cartes (out_channels = M)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.num_attentions == 0:
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)

        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)

        # Activation pour avoir des valeurs positives (similaire à une heatmap)
        # Le code original utilise ELU + 1 pour garantir la positivité tout en gardant un gradient
        x = F.elu(x) + 1

        return x

class AttentionPooling(nn.Module):
    """
    Bloc D : Bilinear Attention Pooling (BAP).
    Combine les features (Texture) et les cartes d'attention.
    Correspond à l'Equation (2) du papier.
    """
    def __init__(self):
        super().__init__()

    def forward(self, features, attentions, norm=2):
        # On s'assure que les dimensions spatiales (H, W) correspondent
        H, W = features.size()[-2:]
        B, M, AH, AW = attentions.size()

        # Si les tailles sont différentes, on redimensionne l'attention pour coller aux features (quelques soucis en lancant des 1ers tests)
        if AH != H or AW != W:
            attentions = F.interpolate(attentions, size=(H, W), mode='bilinear', align_corners=True)

        if norm == 1:
            attentions = attentions + 1e-8

        # Opération bilinéaire : Multiplication élémentaire pondérée
        # Si features a 4 dimensions (B, C, H, W) -> standard
        if len(features.shape) == 4:
            # Einsum pour la multiplication et la somme
            # imjk = Batch, AttentionMaps, H, W
            # injk = Batch, FeatureChannels, H, W
            # -> imn = Batch, AttentionMaps, FeatureChannels
            feature_matrix = torch.einsum('imjk,injk->imn', attentions, features)
        else:
            feature_matrix = torch.einsum('imjk,imnjk->imn', attentions, features)

        # Normalisation 
        if norm == 1:
            w = torch.sum(attentions, dim=(2, 3)).unsqueeze(-1)
            feature_matrix /= w
        if norm == 2:
            feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)
        if norm == 3:
            w = torch.sum(attentions, dim=(2, 3)).unsqueeze(-1) + 1e-8
            feature_matrix /= w

        return feature_matrix

class TextureEnhance(nn.Module):
    """
    Bloc 2 : Texture Enhancement Block.
    Renforce les détails haute fréquence (texture) des couches superficielles
    Version 'v2' du repo github (amélioré depuis le papier) : 100% code github
    """
    def __init__(self, num_features, num_attentions):
        super().__init__()
        # utilise des convolutions groupées par carte d'attention.

        self.output_features = num_features
        self.output_features_d = num_features

        # Extraction initiale
        self.conv_extract = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Dense Block couches (avec groupes = num_attentions pour traiter chaque attention map indépendamment)
        self.conv0 = nn.Conv2d(num_features * num_attentions, num_features * num_attentions, 5, padding=2, groups=num_attentions)
        self.conv1 = nn.Conv2d(num_features * num_attentions, num_features * num_attentions, 3, padding=1, groups=num_attentions)
        self.bn1 = nn.BatchNorm2d(num_features * num_attentions)

        self.conv2 = nn.Conv2d(num_features * 2 * num_attentions, num_features * num_attentions, 3, padding=1, groups=num_attentions)
        self.bn2 = nn.BatchNorm2d(2 * num_features * num_attentions)

        self.conv3 = nn.Conv2d(num_features * 3 * num_attentions, num_features * num_attentions, 3, padding=1, groups=num_attentions)
        self.bn3 = nn.BatchNorm2d(3 * num_features * num_attentions)

        self.conv_last = nn.Conv2d(num_features * 4 * num_attentions, num_features * num_attentions, 1, groups=num_attentions)
        self.bn4 = nn.BatchNorm2d(4 * num_features * num_attentions)
        self.bn_last = nn.BatchNorm2d(num_features * num_attentions)

        self.M = num_attentions

    def cat(self, a, b):
        # Concaténation spécifique pour préserver les groupes d'attention
        B, C, H, W = a.shape
        c = torch.cat([a.reshape(B, self.M, -1, H, W), b.reshape(B, self.M, -1, H, W)], dim=2).reshape(B, -1, H, W)
        return c

    def forward(self, feature_maps, attention_maps):
        B, N, H, W = feature_maps.shape

        # Extraction préliminaire
        feature_maps = self.conv_extract(feature_maps)

        # Calcul du "Global Average" local pour soustraire (Extraction du résidu de texture)
        #  T = f(I) - D
        if isinstance(attention_maps, tuple):
            attention_size = (int(H * attention_maps[0]), int(W * attention_maps[1]))
            attention_maps_resized = 1
        else:
            attention_size = (attention_maps.shape[2], attention_maps.shape[3])
            attention_maps_resized = (torch.tanh(F.interpolate(attention_maps.detach(), (H, W), mode='bilinear', align_corners=True))).unsqueeze(2)

        # Pooling global "soft"
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)

        # Soustraction pour ne garder que la texture (haute fréquence)
        if feature_maps.size(2) > feature_maps_d.size(2):
            feature_maps = feature_maps - F.interpolate(feature_maps_d, (feature_maps.shape[2], feature_maps.shape[3]), mode='nearest')

        #  Multiplication par l'attention (Focus)
        feature_maps = feature_maps.unsqueeze(1)  # B, 1, C, H, W
        if not isinstance(attention_maps_resized, int):
            feature_maps = (feature_maps * attention_maps_resized).reshape(B, -1, H, W)
        else:
            feature_maps = feature_maps.repeat(1, self.M, 1, 1, 1).reshape(B, -1, H, W)

        #  Dense Block Enhancement
        feature_maps0 = self.conv0(feature_maps)

        feature_maps1 = self.conv1(F.relu(self.bn1(feature_maps0), inplace=True))
        feature_maps1_ = self.cat(feature_maps0, feature_maps1)

        feature_maps2 = self.conv2(F.relu(self.bn2(feature_maps1_), inplace=True))
        feature_maps2_ = self.cat(feature_maps1_, feature_maps2)

        feature_maps3 = self.conv3(F.relu(self.bn3(feature_maps2_), inplace=True))
        feature_maps3_ = self.cat(feature_maps2_, feature_maps3)

        # Compression finale
        feature_maps = F.relu(self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_), inplace=True))), inplace=True)

        # Remise en forme (Batch, M_attentions, Channels, H, W)
        feature_maps = feature_maps.reshape(B, -1, N, H, W)

        return feature_maps, feature_maps_d