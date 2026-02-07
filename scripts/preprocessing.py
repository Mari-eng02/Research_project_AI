from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder for categorical features
def encode(df):
    encoded_cat = []
    categorical_features = ['Type', 'Change', 'Urgency', 'Origin']
    for col in categorical_features:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoded_cat.append(df[col].values)
    return np.column_stack(encoded_cat)

# Textual embedding of requirements (normalized)
def embed(df):
    text_model = SentenceTransformer('../models/all-MiniLM-L6-v2', device=device.type)
    text_embeddings = text_model.encode(df['Requirement'].fillna("").tolist(), device=device.type)
    return text_embeddings

# Scaling numeric features
def scale(df, scaler=None, fit=False):
    numeric_features = [col for col in ['N_dependencies', 'Cost', 'Time'] if col in df.columns]
    if fit:
        scaler = MinMaxScaler()
        scaled_train_num = scaler.fit_transform(df[numeric_features])
        return scaled_train_num, scaler
    else:
        scaled_test_num = scaler.transform(df[numeric_features])
        return scaled_test_num
    