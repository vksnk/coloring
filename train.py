import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

if __name__ == "__main__":
    if torch.mps.is_available():
        print('Using mps backend...')
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print('Using cuda backend...')
        device = torch.device("cuda")
    else:
        print('Using cpu backend...')
        device = torch.device("cpu")
