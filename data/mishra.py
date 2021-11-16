import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset 

class Mishra(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_json(file_path, orient="records", lines=True)
        self.texts = self.data['Text']
        self.labels = self.data['Label']
        self.durations = np.array(self.data['Durations'].to_list())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        duration = self.durations[idx]
        label = self.labels[idx]

        return text, label, duration