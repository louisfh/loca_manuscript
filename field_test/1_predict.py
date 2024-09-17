import opensoundscape
# Just for prediction, we used a different version of opensoundscape
assert opensoundscape.__version__ == "0.10.1" 

from opensoundscape.ml.cnn import load_model
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from pathlib import Path
import pandas as pd

model = load_model('2024-02-27_opso-0.10.1_nfwf-v3.0.model', device="cuda:1")

directory = Path("/media/auk/projects/lfh/LOCA22_rotation/synchronized/")
audio_files = list(directory.glob("**/*.WAV"))

# Prediction parameters
num_workers = 8
batch_size = 256

scores = model.predict(audio_files, batch_size=batch_size,num_workers=num_workers,activation_layer=None)
scores.to_csv("./output_data/preds.csv")
