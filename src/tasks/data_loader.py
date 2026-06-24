"""
mante_loader.py
PyTorch DataLoader wrapper with built-in temporal subsampling.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

def load_mante_data(
    npz_path: str, 
    batch_size: int = 64, 
    shuffle: bool = True, 
    subsample_step: int = 1
) -> DataLoader:
    """
    Loads .npz data, applies temporal subsampling, and returns a PyTorch DataLoader.
    
    Args:
        npz_path: Path to the .npz file (e.g., 'data/mante_style/context/train.npz').
        batch_size: Training batch size.
        shuffle: Whether to shuffle the data (True for training, False for validation/testing).
        subsample_step: Stepsize for temporal subsampling. e.g., if dt=1ms and step=10, 
                        the returned tensors act as if dt=10ms.
                        
    Returns:
        PyTorch DataLoader yielding (observations, labels, periods, coherences, contexts).
    """
    data = np.load(npz_path)
    
    # 1. Extract and apply temporal subsampling using NumPy slicing [:, ::step]
    # Shape goes from (Trials, 750, Features) -> (Trials, 75, Features)
    obs = data["observations"][:, ::subsample_step, :]
    labels = data["labels"][:, ::subsample_step]
    periods = data["trial_periods"][:, ::subsample_step]
    
    # Trial-level variables (no temporal dimension)
    cohs = data["coherences"]
    ctxs = data["contexts"]
    
    # 2. Convert to PyTorch tensors
    obs_t = torch.tensor(obs, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    periods_t = torch.tensor(periods, dtype=torch.long)
    cohs_t = torch.tensor(cohs, dtype=torch.float32)
    ctxs_t = torch.tensor(ctxs, dtype=torch.long)
    
    # 3. Build Dataset and Loader
    dataset = TensorDataset(obs_t, labels_t, periods_t, cohs_t, ctxs_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader
