import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
"""
Codes pour évaluer les résultats train val et test
"""


def load_model_weights(model, model_path, device):
    """Charge les poids sauvegardés dans le modèle"""
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def plot_training_results(json_path):
    """Affiche les courbes de perte, acc et AUC depuis le JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(18, 5))

    # Courbe loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss Analysis')
    plt.xlabel('Epochs')
    plt.legend()

    # Courbe AUC
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_auc'], label='Train AUC')
    plt.plot(epochs, history['val_auc'], label='Val AUC')
    plt.title('AUC Analysis')
    plt.xlabel('Epochs')
    plt.legend()

    # Courbe Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Accuracy Analysis')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def run_full_evaluation(model, test_loader, device):
    """
    Calcule les métriques finales et affiche la matrice de confusion
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs['logits'], dim=1)[:, 1]
            preds = torch.argmax(outputs['logits'], dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Métriques sans arrondi pour une précision totale
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n--- TEST SET PERFORMANCE ---")
    print(f"Test Accuracy : {acc}")
    print(f"Test AUC      : {auc}")

    # Affichage de la Matrice de Confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return acc, auc

def visualize_test_attention(model, test_loader, device='cuda', num_samples=4):
    """
    Displays how the M attention heads look at the Test images individually.
    Reprend ta logique de visualisation par tête.
    """
    model.eval()
    # On récupère les données
    inputs, labels = next(iter(test_loader))
    # On limite au nombre de samples demandés
    inputs = inputs[:num_samples].to(device)
    labels = labels[:num_samples]

    with torch.no_grad():
        outputs = model(inputs)
        attentions = outputs['attention_maps'] # [B, M, h, w]

    # Dénormalisation précise pour l'affichage
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(3, 1, 1)
    imgs = (inputs * std + mean).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

    B, M, _, _ = attentions.shape
    # Création d'une grille : B lignes, M+1 colonnes (Original + M têtes)
    fig, axes = plt.subplots(B, M + 1, figsize=(15, B * 3))

    if B == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(B):
        # Colonne 0 : Image Originale
        label_text = 'Fake' if labels[i] == 1 else 'Real'
        axes[i, 0].imshow(imgs[i])
        axes[i, 0].set_title(f"Test ({label_text})")
        axes[i, 0].axis('off')

        # Colonnes 1 à M : Les têtes d'attention
        for k in range(M):
            att_map = attentions[i, k].unsqueeze(0).unsqueeze(0)
            # Redimensionnement à la taille de l'image (380x380)
            att_map = F.interpolate(att_map, size=(380, 380), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            
            axes[i, k+1].imshow(imgs[i])
            axes[i, k+1].imshow(att_map, cmap='jet', alpha=0.4)
            axes[i, k+1].set_title(f"Head {k+1}")
            axes[i, k+1].axis('off')

    plt.tight_layout()
    plt.show()



'''import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.model import MultiAttentionNet

def plot_training_results(json_path):
    """
    Reads the JSON log and plots Loss, Accuracy, and AUC.
    No approximations: uses raw values from the file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))
    
    # 1. Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='.')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='.')
    plt.title(f"Loss - {os.path.basename(json_path)}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Accuracy & AUC Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='s')
    plt.plot(epochs, history['val_auc'], label='Val AUC', linestyle='--', color='green')
    plt.title("Accuracy & AUC")
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

def load_model_weights(model_path, m_heads, device='cuda'):
    """
    Rebuilds the model and loads the specific .pth weights.
    """
    # Recreate the skeleton (defaulting to b0 as we discussed for tests)
    model = MultiAttentionNet(model_name='efficientnet_b0', M=m_heads)
    
    # Load the weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Loaded weights: {os.path.basename(model_path)}")
    return model

def visualize_test_attention(model, test_loader, device='cuda', num_samples=4):
    """
    Displays how the M attention heads look at the Test images.
    """
    model.eval()
    inputs, _ = next(iter(test_loader))
    inputs = inputs[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(inputs)
        attentions = outputs['attention_maps'] # [B, M, h, w]

    # Denormalize for plotting
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(3, 1, 1)
    imgs = (inputs * std + mean).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

    B, M, _, _ = attentions.shape
    fig, axes = plt.subplots(B, M + 1, figsize=(15, B * 3))

    for i in range(B):
        # Original
        axes[i, 0].imshow(imgs[i])
        axes[i, 0].set_title("Test Image")
        axes[i, 0].axis('off')

        # Each attention head
        for k in range(M):
            att_map = attentions[i, k].unsqueeze(0).unsqueeze(0)
            att_map = F.interpolate(att_map, size=(380, 380), mode='bilinear').squeeze().cpu().numpy()
            
            axes[i, k+1].imshow(imgs[i])
            axes[i, k+1].imshow(att_map, cmap='jet', alpha=0.4)
            axes[i, k+1].set_title(f"Head {k+1}")
            axes[i, k+1].axis('off')

    plt.tight_layout()
    plt.show()

def plot_test_confusion_matrix(model, test_loader, device='cuda'):
    """
    Runs the model on the entire test set and plots the Confusion Matrix.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs['logits'], dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    '''