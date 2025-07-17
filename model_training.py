import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from msa_embedding import msa_to_embedding

class SimpleProteinTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(embed_dim, 3)  # Exemplo: 3 classes estruturais

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pooling
        return self.classifier(x)

def train_model(msa_emb_path: str):
    data = msa_to_embedding(msa_emb_path)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels = torch.randint(0, 3, (data.shape[0],))  # Fake labels só p/ protótipo

    dataset = TensorDataset(data_tensor, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = SimpleProteinTransformer(input_dim=data.shape[2])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}: loss {loss.item():.4f}")

# Exemplo
if __name__ == "__main__":
    train_model("msa_output/output.a3m")
