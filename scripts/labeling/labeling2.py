import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch.nn import Linear, MSELoss
from torch.optim import Adam
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scripts.preprocessing import encode, embed

import torch.nn.functional as F
import numpy as np
import random

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# LOAD THE REQUIREMENTS DATASETS
df_train = pd.read_csv("../../dataset/train_labeled.csv")
df_test = pd.read_csv("../../dataset/test_labeled.csv")

# Concatenation of training and test set (temporary)
df = pd.concat([df_train,df_test], ignore_index=True).reset_index(drop=True)

# Preprocessing
text_embeddings = torch.tensor(embed(df), device=device)
encoded_cat = encode(df)

# Construction of pseudo-labels
sim_matrix = cosine_similarity(text_embeddings.cpu().numpy())
threshold = 0.75
pseudo_labels = (sim_matrix > threshold).sum(axis=1) - 1
df['pseudo_N_dependencies'] = pseudo_labels

# Construction of the graph
categorical_features = torch.tensor(df[['Type', 'Change', 'Urgency', 'Origin']].values, dtype=torch.float, device=device)
node_features = torch.cat([text_embeddings, categorical_features], dim=1)

edge_index = []
for i in range(len(sim_matrix)):
    for j in range(len(sim_matrix)):
        if i != j and sim_matrix[i][j] > threshold:
            edge_index.append([i, j])       # adds an arc from i to j if the two requirements are semantically similar

# Converts the list to a PyTorch tensor, as required by PyTorch Geometric
edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()

# Target: N_dependencies
y = torch.tensor(df['pseudo_N_dependencies'].values, dtype=torch.float, device=device)
data = Data(x=node_features, edge_index=edge_index, y=y)

# Model GNN definition
class GNNRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.lin(x).squeeze()

# Model Training
model = GNNRegressor(in_channels=data.x.shape[1], hidden_channels=64).to(device)
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = MSELoss()

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Model Evaluation
model.eval()
with torch.no_grad():
    predictions = model(data.x, data.edge_index).cpu().numpy()

predictions = np.round(predictions).astype(int)

# Predictions analysis
print("Predictions statistics:")
print(f"Min: {predictions.min()}")
print(f"Max: {predictions.max()}")
print(f"Mean: {predictions.mean():.2f}")

# Saving of the predictions in the original datasets
df['N_dependencies'] = predictions
df_train['N_dependencies'] = df.loc[:len(df_train) - 1, 'N_dependencies'].values
df_test['N_dependencies'] = df.loc[len(df_train):, 'N_dependencies'].values

df_train.to_csv("../../dataset/train_labeled.csv", index=False)
df_test.to_csv("../../dataset/test_labeled.csv", index=False)

print("✅ Dependencies predictions saved in train_labeled.csv")
print("✅ Dependencies predictions saved in test_labeled.csv")

torch.save(model.state_dict(), "../../models/gnn_model.pt")


