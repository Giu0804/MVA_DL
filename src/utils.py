import json
import os

"""
Enregistrement des métriques pour évaluation ultérieure
Enregistrement des poids du meilleur modèle obtenu pour chaque expérience pour évaluer sur test set 
"""

class ExperimentJSONLogger:
    def __init__(self, filepath, config):
        self.filepath = filepath
        self.data = {
            "config": config,
            "history": {
                "train_loss": [],
                "train_acc": [],
                "train_auc": [],
                "val_loss": [],
                "val_acc": [],  
                "val_auc": []
            }
        }
        # Création du dossier logs s'il n'existe pas
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._save()

    def _save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=4)

    def log_epoch(self, metrics_dict):
        """
        Reçoit le dictionnaire de metrics et met à jour le JSON.
        """
        for key, value in metrics_dict.items():
            # On vérifie si la clé existe dans notre dictionnaire history
            if key in self.data["history"]:
                self.data["history"][key].append(float(value))
        
        self._save()
        print(f"Metrics saved to {self.filepath}")