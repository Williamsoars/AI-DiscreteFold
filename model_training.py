import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from msa_embedding import esm_embedding_from_fasta
from pdb_structure import extract_ca_coordinates
import numpy as np

class FoldingRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 3)]  # 3D coord
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_folding_model(fasta_path: str, pdb_path: str, epochs=20):
    # Embeddings e coordenadas
    embeddings = esm_embedding_from_fasta(fasta_path)  # shape: (1, d)
    coords = extract_ca_coordinates(pdb_path)          # shape: (n_res, 3)

    if embeddings.shape[0] != coords.shape[0]:
        raise ValueError(f"Tamanho do embedding ({embeddings.shape[0]}) e das coords ({coords.shape[0]}) não coincidem")

    x_tensor = embeddings.float()
    y_tensor = torch.tensor(coords, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = FoldingRegressor(input_dim=embeddings.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    return model



