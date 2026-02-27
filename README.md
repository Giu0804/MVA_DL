# Initial set up

We recommend using [Onyxia](https://datalab.sspcloud.fr/), which provides storage resources and GPUs, or **Colab Pro**.  
The code will not run completely on the free version of Colab.

If you are working locally and have access to GPUs, we recommend creating a virtual environment:

**Windows:**  
`python -m venv nom_env`  
then  
`nom_env\Scripts\Activate.ps1`

**Linux:**  
`python -m venv nom_env`  
then  
`source nom_env/bin/activate`

To install the dependencies, run the following command:

`
pip install -r requirements.txt
`

# Data Download

Since the datasets are large, we were unable to include them in our GitHub repository. 

The data can be downloaded from **Kaggle**. We used the following image datasets:  
- [Forensics++ Videos Cropped Faces](https://www.kaggle.com/datasets/debajyatidey/faceforensics-videos-cropped-faces/data)  
- [CelebDF_V2 (image Dataset)](https://www.kaggle.com/datasets/pranabr0y/celebdf-v2image-dataset/data)

## Locally: 
Simply download the ZIP file and extract the data in the data Folder.
For Forensics++, there will be two folders: 'fake' and 'real'.  
Create a 'Forensics++' folder in the dat folder and put them inside. 

## On Onyxia:
The procedure is as follows:

1. Get an API token on Kaggle.
2. Then run:

```bash
mkdir -p ~/.kaggle  

nano ~/.kaggle/kaggle.json
# Add the following content:
{ "username": "your_username", "key": "your_key" }

chmod 600 ~/.kaggle/kaggle.json
```

To download, Forensics++ dataset, run: 
`kaggle datasets download -d debajyatidey/faceforensics-videos-cropped-faces -p ~/work/MVA_DL/data --unzip`

To download, Celeb DF dataset, run: 
`kaggle datasets download -d pranabr0y/celebdf-v2image-dataset -p ~/work/MVA_DL/data --unzip`

For Forensics++, there will be two folders: 'fake' and 'real'.  
Create a 'Forensics++' folder in the data folder and put them inside.  

# Models 

The models (their weights) are stored in the **"models"** folder.  
In the **"logs"** folder, you can check their performance (loss, accuracy, etc.) stored in JSON files.

If you want to run the experiments, you need to run the `experiments.py` file.

For persistence, you can use **tmux**.
