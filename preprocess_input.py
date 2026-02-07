import pandas as pd
import numpy as np
from scripts.preprocessing import *
from sklearn.decomposition import PCA
import joblib

pca_emb = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")

def preprocess_input(row_dict):
    # dataframe with only one row
    df = pd.DataFrame([row_dict])
    # preprocessing (scaling, encoding, embedding)
    scaled_num = scale(df, scaler=scaler)
    encoded_cat = encode(df)
    emb = embed(df)
    emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    emb_reduced = pca_emb.transform(emb_norm)
    # concatenation
    X = np.hstack([emb_reduced * 0.5, encoded_cat.astype(float), scaled_num])
    return X[0]
