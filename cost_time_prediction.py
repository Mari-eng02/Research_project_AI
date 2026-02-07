import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from preprocess_input import preprocess_input

df_train = pd.read_csv("dataset/train_labeled.csv")
df_processed = df_train.apply(preprocess_input, axis=1).apply(pd.Series)

X_processed = df_processed.iloc[:, :-2].values
y_train = df_train[["Cost", "Time"]].values

model =  RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_processed, y_train)

joblib.dump(model, "models/regressor_cost_time.pkl")


