#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import torch
import torch.nn as nn
import itertools
from src.dataset import create_dataloaders
from src.model import MultiAttentionNet
from src.loss import RegionalIndependenceLoss
from src.agda import AGDA
from src.trainer import Trainer
from src.utils import ExperimentJSONLogger


# In[2]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# In[3]:


# --- GRID SEARCH CONFIG ---
# On réduit à 12 combinaisons pour tenir les délais
lrs = [1e-4, 1e-3]
batch_sizes = [16, 32]
dropouts = [0, 0.1, 0.3]
M_heads = [4, 8] # Priorité au M=4 du papier

# Paramètres fixes
base_config = {
    "epochs": 12,
    "patience": 5,
    "frames_per_video": 10,
    "max_videos": 300, # 100 vidéos total (50R/50F)
    "agda_prob": 0.5,
    "weight_decay": 1e-6
}

# Génération de la grille
grid = list(itertools.product(lrs, batch_sizes, dropouts, M_heads))


# In[ ]:


# --- GRID SEARCH EXECUTION ---

for lr, bs, dropout, M in grid:
    # 1. Définition du nom unique pour ce run
    run_name = f"lr{lr}_bs{bs}_d{dropout}_M{M}_ep{base_config['epochs']}"
    log_path = f"logs/{run_name}.json"
    model_path = f"models/{run_name}.pth"

    # Sécurité Onyxia : Reprise si crash
    if os.path.exists(log_path):
        print(f"Skip: {run_name} already done")
        continue

    print(f"Experiment: {run_name}")

    # --- A. Data (Recréé à chaque fois car le Batch Size change) ---
    dataloaders, sizes = create_dataloaders(
        root_dir="data/Forensics++/",
        batch_size=bs,
        frames_per_video=base_config["frames_per_video"],
        max_videos=base_config["max_videos"]
    )

    # --- B. Model & Components (M et Dropout changent) ---
    model = MultiAttentionNet(
        model_name="efficientnet_b0",
        M=M,
        dropout_rate=dropout
    ).to(DEVICE)

    # Paramètres RIL conformes au papier
    criterion_ril = RegionalIndependenceLoss(
        M=M, 
        num_features_d=model.feat_dim_texture, 
        m_out=0.2, 
        m_in_real=0.05, 
        m_in_fake=0.1
    ).to(DEVICE)

    criterion_ce = nn.CrossEntropyLoss()
    agda = AGDA(p=base_config["agda_prob"])

    # L'optimizer doit être lié aux nouveaux paramètres du nouveau modèle !
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=base_config["weight_decay"]
    )

    # --- C. Logging ---
    current_config = {**base_config, "lr": lr, "bs": bs, "dropout": dropout, "M": M}
    logger = ExperimentJSONLogger(filepath=log_path, config=current_config)

    # --- D. Training ---
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        criterion_ce=criterion_ce,
        criterion_ril=criterion_ril,
        optimizer=optimizer,
        agda=agda,
        device=DEVICE
    )

    # Lancement de l'entraînement avec Early Stopping
    trainer.fit(
        num_epochs=base_config["epochs"],
        model_save_path=model_path,
        log_callback=logger.log_epoch,
        patience=base_config["patience"]
    )

    # 
    del model, trainer, dataloaders, optimizer, criterion_ril
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Run {run_name} finished")


# In[ ]:




