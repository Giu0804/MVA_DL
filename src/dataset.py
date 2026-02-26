import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

"""
Data/Forensics++ -->
- fake/id_id_crops/frame.png
- real/id_crops/frame.png

trade off: distribution équilibrée entre les fake et les real et nombre limité de données pour faire tourner le code
revoir construction du dataset initial (forensics comment ca marche) mais en gros les id doivent etre différents pour éviter des biais 
"""

class ForensicsDataset(Dataset):
    """
    Dataset qui extrait N vidéos et M images par vidéo
    """
    def __init__(self, video_paths, labels, frames_per_video=10, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.samples = []

        for path, label in zip(video_paths, labels):
            # Récupération des images dans le dossier vidéo
            images = sorted([
                os.path.join(path, f) 
                for f in os.listdir(path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            
            if len(images) >= frames_per_video:
                # Sélection uniforme des frames
                idx = torch.linspace(0, len(images) - 1, frames_per_video).long()
                for i in idx:
                    self.samples.append((images[i], label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def create_dataloaders(root_dir, batch_size=16, frames_per_video=10, max_videos=None):
    """
    Crée des loaders avec un split 80/10/10 par ID et 50/50 Real/Fake
    """
    categories = {'real': 0, 'fake': 1}
    # Structure : data_by_id[label][pid] = [liste des dossiers]
    data_by_id = {0: {}, 1: {}}

    # 1. Collecte et filtrage initial (max_videos)
    for cat, label in categories.items():
        cat_path = os.path.join(root_dir, cat)
        if not os.path.exists(cat_path):
            print(f"Warning: Folder {cat_path} not found")
            continue
            
        video_folders = sorted([
            d for d in os.listdir(cat_path) 
            if os.path.isdir(os.path.join(cat_path, d))
        ])
        
        # Imposer une limite, risque de jamais finir avec toutes les données
        if max_videos:
            random.seed(42)
            random.shuffle(video_folders)
            video_folders = video_folders[:max_videos]

        for f in video_folders:
            # Extraction du PID (ex: "001" de "001_v1_crops")
            pid = f.split('_')[0]
            if pid not in data_by_id[label]:
                data_by_id[label][pid] = []
            data_by_id[label][pid].append(os.path.join(cat_path, f))

    # 2. Split équilibré : On split les IDs pour CHAQUE catégorie
    splits = {'train': [], 'val': [], 'test': []}
    
    random.seed(42) # Pour la reproductibilité du split
    
    for label in [0, 1]:
        ids = sorted(list(data_by_id[label].keys()))
        random.shuffle(ids)
        
        n = len(ids)
        tr_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        # Répartition des IDs
        phase_map = {
            'train': ids[:tr_end],
            'val': ids[tr_end:val_end],
            'test': ids[val_end:]
        }
        
        for phase, p_ids in phase_map.items():
            for pid in p_ids:
                for v_path in data_by_id[label][pid]:
                    splits[phase].append((v_path, label))

    # 3. Configuration des Transforms comme dans le papier
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Création des DataLoaders
    loaders = {}
    sizes = {}
    
    for phase in ['train', 'val', 'test']:
        if not splits[phase]:
            continue
            
        # Mélange final pour mixer les Real et Fake dans les batchs
        random.shuffle(splits[phase])
        
        v_paths, v_labels = zip(*splits[phase])
        dataset = ForensicsDataset(v_paths, v_labels, frames_per_video, transform)
        
        loaders[phase] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(phase == 'train'), 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        sizes[phase] = len(dataset)
        
        # Affichage des statistiques : check que c est bon 
        n_real = sum(1 for l in v_labels if l == 0)
        n_fake = sum(1 for l in v_labels if l == 1)
        print(f"Phase '{phase}': {len(v_paths)} vidéos ({n_real}R/{n_fake}F), {len(dataset)} images.")

    return loaders, sizes