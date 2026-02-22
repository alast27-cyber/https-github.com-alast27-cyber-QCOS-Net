import numpy as np
import torch
from torch.utils.data import Dataset

class QCOSDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        """
        Generates synthetic training data for the OS Kernel.
        Inputs: [Context (C), Energy (E)]
        Labels: 1 (Act) or -1 (Rest/Veto) for Quantum Hinge Loss
        """
        data = []
        labels = []
        
        for _ in range(self.num_samples):
            c = np.random.rand() # 0.0 to 1.0
            e = np.random.rand() # 0.0 to 1.0
            
            # --- THE GROUND TRUTH LOGIC (The Teacher) ---
            # Rule 1: High Energy (> 0.6) is always REST (-1)
            if e > 0.6:
                label = -1.0
            # Rule 2: High Context (> 0.5) with Low Energy is ACT (1)
            elif c > 0.5:
                label = 1.0
            # Rule 3: Low Context is REST (-1)
            else:
                label = -1.0
                
            data.append([c, e])
            labels.append(label)
            
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
