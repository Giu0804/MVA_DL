import os
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm
import json

"""
Entrainement du modèle
Métriques évaluées : Loss et Accuracy + AUC (métrique du papier) sur train et val
"""


class Trainer:
    def __init__(self, model, dataloaders, criterion_ce, criterion_ril, optimizer, agda=None, device='cuda'):
        """
        Gère l'entraînement et la validation du modèle.
        """
        self.device = device
        self.model = model.to(self.device)
        self.dataloaders = dataloaders
        self.criterion_ce = criterion_ce
        self.criterion_ril = criterion_ril
        self.optimizer = optimizer
        self.agda = agda
        
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': []
        }

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        for inputs, labels in self.dataloaders['train']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # 1er passage
            outputs = self.model(inputs)
            
            loss_ce = self.criterion_ce(outputs['logits'], labels)
            loss_ril = self.criterion_ril(outputs['texture_raw'], outputs['attention_maps'], labels)
            loss = loss_ce + loss_ril

            # -2eme passage agda
            if self.agda is not None:
                inputs_aug, _ = self.agda(inputs, outputs['attention_maps'].detach())
                
                if inputs_aug is not None:
                    outputs_aug = self.model(inputs_aug)
                    loss_ce_aug = self.criterion_ce(outputs_aug['logits'], labels)
                    
                    # On fait la moyenne des deux CE pour garder l'équilibre avec la RIL
                    loss = (loss_ce + loss_ce_aug) / 2.0 + loss_ril

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs['logits'], dim=1)[:, 1]
            preds = torch.argmax(outputs['logits'], dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

        epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        try:
            epoch_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            epoch_auc = 0.5 

        return epoch_loss, epoch_acc, epoch_auc

    def val_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in self.dataloaders['val']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss_ce = self.criterion_ce(outputs['logits'], labels)
                loss_ril = self.criterion_ril(outputs['texture_raw'], outputs['attention_maps'], labels)
                loss = loss_ce + loss_ril

                running_loss += loss.item() * inputs.size(0)
                probs = torch.softmax(outputs['logits'], dim=1)[:, 1]
                preds = torch.argmax(outputs['logits'], dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / len(self.dataloaders['val'].dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        try:
            epoch_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            epoch_auc = 0.5

        return epoch_loss, epoch_acc, epoch_auc

    def fit(self, num_epochs, model_save_path, log_callback=None, patience=5):
        best_val_auc = 0.0
        best_val_auc = -1.0 
        best_val_loss = float('inf') 
        epochs_no_improve = 0        # Compteur pour l'early stopping

        for epoch in range(num_epochs):
            start_time = time.time()

            # 1. Entraînement et Validation
            train_loss, train_acc, train_auc = self.train_epoch()
            val_loss, val_acc, val_auc = self.val_epoch()

            # 2. Mise à jour de l'Alpha (RIL Decay) à chaque fin d'époque
            if hasattr(self.criterion_ril, 'update_alpha'):
                self.criterion_ril.update_alpha(decay=0.9)

            metrics = {
                'train_loss': train_loss, 'train_acc': train_acc, 'train_auc': train_auc,
                'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc
            }
            
            for k, v in metrics.items():
                self.history[k].append(v)

            epoch_time = time.time() - start_time
            msg = f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time}s | "
            msg += f"Train Loss: {train_loss} AUC: {train_auc} | "
            msg += f"Val Loss: {val_loss} AUC: {val_auc}"

            # 3. Sauvegarde du meilleur modèle basé sur l'AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(self.model.state_dict(), model_save_path)
                msg += " <- Best model saved"

            print(msg)

            # Logique de l'early stopping  patience sur la val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0 
            else:
                epochs_no_improve += 1
                print(f"Patience: {epochs_no_improve}/{patience}")

            if log_callback is not None:
                log_callback(metrics)

            # 5. Condition d'arrêt
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch: {epoch+1} ")
                break

        return self.history


