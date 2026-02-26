import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules import AttentionPooling 

# REVOIR DIFF ET LIEN AVEC LE GITHUB

class RegionalIndependenceLoss(nn.Module):
    """
    Loss RIL 
    """
    def __init__(self, M, num_features_d, alpha=0.05, m_out=0.2, m_in_real=0.05, m_in_fake=0.1):
        super().__init__()
        self.M = M
        self.num_features_d = num_features_d
        self.alpha = alpha
        self.m_out = m_out           # Marges inter-classe 
        self.m_in_real = m_in_real   # Marges intra-classe 
        self.m_in_fake = m_in_fake
        
        self.atp = AttentionPooling()
        
        # Buffers pour les centres (mis à jour manuellement, pas par l'optimiseur)
        self.register_buffer('feature_centers', torch.zeros(M, num_features_d))

    def forward(self, feature_map_d, attentions, y):
        B = y.size(0)
        device = feature_map_d.device
        
        #  Extraction des vecteurs sémantiques V 
        # On utilise norm=1 pour coller au Normalized Average Pooling du papier
        V = self.atp(feature_map_d, attentions, norm=1) # [B, M, D]
        
        #  Mise à jour des centres
        if self.training:
            with torch.no_grad():
                # On met à jour les centres avec le momentum alpha
                # V_mean: moyenne des caractéristiques du batch actuel
                V_mean = torch.mean(V, dim=0) 
                self.feature_centers.copy_(self.feature_centers * (1 - self.alpha) + V_mean * self.alpha)

        #  INTRA-CLASS LOSS
        # Distance entre les caractéristiques et leur centre respectif
        centers = self.feature_centers.unsqueeze(0) # [1, M, D]
        dist_sq = torch.sum((V - centers)**2, dim=-1) # [B, M] (L2 norm au carré)
        
        # Attribution des marges selon le label 
        m_in = torch.where(y == 0, self.m_in_real, self.m_in_fake).view(-1, 1)
        intra_loss = torch.mean(F.relu(dist_sq - m_in))

        # INTER-CLASS LOSS
        # On veut que les centres soient éloignés d'au moins m_out
        inter_loss = 0
        c = self.feature_centers
        for i in range(self.M):
            for j in range(i + 1, self.M):
                dist_centers = torch.sum((c[i] - c[j])**2)
                inter_loss += F.relu(self.m_out - dist_centers)
        
        # Normalisation par le nombre de paires
        num_pairs = self.M * (self.M - 1) / 2
        inter_loss = inter_loss / num_pairs if num_pairs > 0 else 0
        
        return intra_loss + inter_loss
