import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# Create DataLoaders
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


import torch
from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    def __init__(self, dataX, dataY, max_samples=None):
        """
        Initializes the dataset.
        
        Parameters:
            dataX (numpy array): Input data, shape (samples, features).
            dataY (numpy array): Labels, shape (samples,).
            max_samples (int, optional): Maximum number of samples to include in the dataset.
        """
        if max_samples is not None:
            dataX = dataX[:max_samples]
            dataY = dataY[:max_samples]
        
        self.dataX = torch.tensor(dataX, dtype=torch.float32)  # Convert to torch tensor
        self.dataY = torch.tensor(dataY, dtype=torch.long)     # Convert to torch tensor
    
    def __len__(self):
        return len(self.dataY)
    
    def __getitem__(self, idx):
        return self.dataX[idx], self.dataY[idx]
