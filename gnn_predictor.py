import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import SAGEConv
from torch.nn import Linear
import torch.nn.functional as F

from scripts.preprocessing import embed, encode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def predict_dependencies(requirement_row, model, existing_df, device=device, similarity_threshold=0.75):
    new_embedding = torch.tensor(embed(pd.DataFrame([requirement_row])), dtype=torch.float, device=device)

    new_cat = encode(pd.DataFrame([requirement_row]))
    new_cat = torch.tensor(new_cat, dtype=torch.float, device=device)

    new_node_feat = torch.cat([new_embedding, new_cat], dim=1)

    existing_embeddings = torch.tensor(embed(existing_df), dtype=torch.float, device=device)
    existing_cat = encode(existing_df)
    existing_cat = torch.tensor(existing_cat, dtype=torch.float, device=device)
    existing_node_feats = torch.cat([existing_embeddings, existing_cat], dim=1)

    all_node_features = torch.cat([new_node_feat, existing_node_feats], dim=0)

    sim_matrix = cosine_similarity(existing_embeddings.cpu().numpy())

    existing_edges = []
    num_existing = len(existing_df)
    for i in range(num_existing):
        for j in range(i + 1, num_existing):
            if sim_matrix[i][j] > similarity_threshold:
                existing_edges.append([i + 1, j + 1])
                existing_edges.append([j + 1, i + 1])

    sim_new = cosine_similarity(new_embedding.cpu().numpy(), existing_embeddings.cpu().numpy())[0]

    new_edges = [[0, i + 1] for i, s in enumerate(sim_new) if s > similarity_threshold]
    new_edges += [[i + 1, 0] for i, s in enumerate(sim_new) if s > similarity_threshold]

    if not new_edges:
        top_match = int(np.argmax(sim_new))
        new_edges = [[0, top_match + 1], [top_match + 1, 0]]

    all_edges = new_edges + existing_edges
    edge_index = torch.tensor(all_edges, dtype=torch.long, device=device).t().contiguous()


    model.eval()
    with torch.no_grad():
        out = model(all_node_features, edge_index)
        prediction = out[0].item()

    return int(round(prediction)), new_embedding, new_cat



