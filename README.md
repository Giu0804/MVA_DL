- Load data : Get API token on Kaggle

  
  1) A la racine (dans work): mkdir -p ~/.kaggle
  2) nano ~/.kaggle/kaggle.json
- {
  "username": "nom_token",
  "key": "key_token"
} 
3) chmod 600 ~/.kaggle/kaggle.json


- pip install kaggle (mis dans requirements normalement)

- pip install -r requirements.txt

- kaggle datasets download -d pranabr0y/celebdf-v2image-dataset -p ~/work/data --unzip (pas utilisé pour le moment)

- kaggle datasets download -d debajyatidey/faceforensics-videos-cropped-faces -p ~/data --unzip

- le 2eme telechargement crée 2 dossiers fake et real, je les ai mis dans un dossier nommé Forensics++

- pour lancer le notebook sur le terminal, transformer en py file

- nohup python experiments.py > output.log 2>&1 &  pour lancer le code

- ps aux | grep python voir si ca tourne

-  output log a la racine poru check si ca va
