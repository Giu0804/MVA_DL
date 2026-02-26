import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class AGDA(nn.Module):
    """
    Attention Guided Data Augmentation (Block F)
    """
    def __init__(self, p=0.5, sigma=7.0):
        """
        Args:
            p: Probabilité d'appliquer l'AGDA sur un batch donné.
            sigma: Intensité du flou Gaussien l(e papier recommande sigma=7)
        """
        super(AGDA, self).__init__()
        self.p = p

        # Le noyau (kernel) doit être impair (conseil gemini)
        # Pour sigma=7, un noyau de 21 ou 23 couvre bien la gaussienne (3*sigma)
        self.blur_transform = transforms.GaussianBlur(kernel_size=23, sigma=sigma)

    def forward(self, images, attention_maps):
        if not self.training or torch.rand(1) > self.p:
            return images, None

        B, M, h, w = attention_maps.size()
        H, W = images.size(2), images.size(3)

        # sélection aléatoire
        selected_indices = torch.randint(0, M, (B,), device=images.device)
        indices_view = selected_indices.view(B, 1, 1, 1).expand(-1, 1, h, w)
        selected_maps = torch.gather(attention_maps, 1, indices_view)

        # normalisation Min-Max 
        flat_maps = selected_maps.view(B, -1)
        min_vals = flat_maps.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_vals = flat_maps.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        mask = (selected_maps - min_vals) / (max_vals - min_vals + 1e-8)
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=True)

        # Création de l'image dégradée Id (avec un Resize factor 0.3)
        # On réduit l'image à 30% de sa taille
        low_res_size = (int(H * 0.3), int(W * 0.3))
        images_low = F.interpolate(images, size=low_res_size, mode='bilinear', align_corners=True)

        # On applique le flou sur la version basse résolution
        blurred_low = self.blur_transform(images_low)

        # On remonte à la taille originale (380x380)
        Id = F.interpolate(blurred_low, size=(H, W), mode='bilinear', align_corners=True)

        # 4. Mixage final (equation papier)
        images_aug = images * (1 - mask) + Id * mask

        return images_aug, selected_indices
