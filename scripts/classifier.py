import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from preprocessing import *

# Datasets loading
df_train = pd.read_csv("../dataset/train_labeled.csv")
df_test = pd.read_csv("../dataset/test_labeled.csv")

print(df_train.shape)
print(df_test.shape)

# Features
X_train = df_train.drop(columns=["Priority"], axis=1)
X_test = df_test.drop(columns=["Priority"], axis=1)

# Preprocessing 
scaled_train_num, scaler = scale(X_train, fit=True)
scaled_test_num = scale(X_test, scaler=scaler)

embeddings_train = embed(X_train)
norm_embeddings_train = embeddings_train / np.linalg.norm(embeddings_train, axis=1, keepdims=True)

embeddings_test = embed(X_test)
norm_embeddings_test = embeddings_test / np.linalg.norm(embeddings_test, axis=1, keepdims=True)

encoded_train_cat = encode(X_train)
encoded_test_cat = encode(X_test)

# Reduction of embeddings
pca_emb = PCA(n_components=0.80, random_state=42)
embeddings_train_reduced = pca_emb.fit_transform(norm_embeddings_train)
embeddings_test_reduced = pca_emb.transform(norm_embeddings_test)

# Concatenation
X_train_full = np.hstack([embeddings_train_reduced*0.5, encoded_train_cat.astype(float), scaled_train_num])
X_test_full = np.hstack([embeddings_test_reduced*0.5, encoded_test_cat.astype(float), scaled_test_num])

print(X_train_full.shape)
print(X_test_full.shape)

#######################################################################################################################

# Clustering (KMeans)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters_train = kmeans.fit_predict(X_train_full)
clusters_test = kmeans.predict(X_test_full)

# Silhouette score
print(f"\nSilhouette score training set: {silhouette_score(X_train_full, clusters_train):.3f}")
print(f"Silhouette score test set: {silhouette_score(X_test_full, clusters_test):.3f}")


# 2D Clustering Visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_embedded = tsne.fit_transform(X_train_full)

plt.figure(figsize=(8,6))
plt.scatter(tsne_embedded[:, 0], tsne_embedded[:, 1], c=clusters_train, cmap='tab10', s=2)
plt.title('t-SNE of Requirements Clustering')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True)
plt.colorbar(label='Cluster')
plt.show()

# Assign priority to each cluster, based on medium urgency
urgency_encoded = encoded_train_cat[:, 2]

temp_df = pd.DataFrame({
    'Cluster': clusters_train,
    'Urgency_encoded': urgency_encoded
})

cluster_urgency = temp_df.groupby('Cluster')['Urgency_encoded'].mean()
cluster_order = cluster_urgency.sort_values(ascending=False).index

priority_levels = ['high', 'medium', 'low']
cluster_to_priority = {cluster_id: priority for cluster_id, priority in zip(cluster_order, priority_levels)}

print("\nUrgency mean and assigned priority for each cluster:")
for cluster_id in cluster_order:
    urgency_mean = cluster_urgency[cluster_id]
    priority = cluster_to_priority[cluster_id]
    print(f"Cluster {cluster_id}: Urgency mean = {urgency_mean:.2f} → Priority = {priority}")


# Apply priority to the dataframe
df_train['Priority'] = pd.Series(clusters_train, index=df_train.index).map(cluster_to_priority)
df_test['Priority'] = pd.Series(clusters_test, index=df_test.index).map(cluster_to_priority)

# Saving data in the original dataset
df_train.to_csv("../dataset/train_labeled.csv", index=False)
df_test.to_csv("../dataset/test_labeled.csv", index=False)

print("\n✅ Priorities saved in the datasets")


######################################################## Main ########################################################

# Target Encoding
y_train = df_train["Priority"]
y_test = df_test["Priority"]

target_encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
y_train = target_encoder.fit_transform(y_train.to_frame()).ravel()
y_test = target_encoder.transform(y_test.to_frame()).ravel()


# Correlation matrix
X_combined = np.vstack([X_train_full, X_test_full])
y_combined = np.concatenate([y_train, y_test]).reshape(-1,1)

embedding_cols_reduced = [f'emb_PC{i+1}' for i in range(embeddings_train_reduced.shape[1])]
encoded_cols = ['Type', 'Change', 'Urgency', 'Origin']
scaled_cols = ['N_dependencies', 'Cost', 'Time']
feature_names = embedding_cols_reduced + encoded_cols + scaled_cols

np_combined = np.hstack([X_combined, y_combined])
df_combined = pd.DataFrame(np_combined, columns=feature_names + ['Priority'])

corr_matrix = df_combined.corr()

# Aggregate embeddings
other_cols = encoded_cols + scaled_cols + ['Priority']
grouped_corr = pd.DataFrame(index=['Embeddings'] + other_cols, columns=['Embeddings'] + other_cols)

# Embeddings vs embeddings
grouped_corr.loc['Embeddings', 'Embeddings'] = corr_matrix.loc[embedding_cols_reduced, embedding_cols_reduced].values.mean()

# Embeddings vs other features
for col in other_cols:
    grouped_corr.loc['Embeddings', col] = corr_matrix.loc[embedding_cols_reduced, col].values.mean()
    grouped_corr.loc[col, 'Embeddings'] = corr_matrix.loc[col, embedding_cols_reduced].values.mean()

# Other features vs other features
for col1 in other_cols:
    for col2 in other_cols:
        grouped_corr.loc[col1, col2] = corr_matrix.loc[col1, col2]

# Convert to float
grouped_corr = grouped_corr.astype(float)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(grouped_corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Features Correlation Heatmap (Embeddings Aggregated)")
plt.tight_layout()
plt.show()

#######################################################################################################################

# Random Forest Classification
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

'''
# Decision Tree Classification
model = DecisionTreeClassifier(
    criterion='entropy',
    min_samples_split=2,
    min_samples_leaf=2,
    max_depth=5,
    random_state=42
)

# Rule-based Classification
def rule_based_priority(row):
    if row['Urgency'] == 'soon' and row['N_dependencies'] > 2:
        return 'high'
    elif row['Urgency'] == 'later' and row['N_dependencies'] <= 2:
        return 'low'
    else:
        return 'medium'

y_pred = df_test[['Urgency', 'N_dependencies']].apply(rule_based_priority, axis=1)
y_pred = target_encoder.transform(y_pred.to_frame()).ravel()    # rule-based predictions encoding


############### Ablation Study ###############

# A1: Only Embeddings
X_train_emb = embeddings_train_reduced * 0.5
X_test_emb = embeddings_test_reduced * 0.5

model.fit(X_train_emb, y_train)
y_pred = model.predict(X_test_emb)

# A2: No Embeddings
X_train_no_emb = np.hstack([encoded_train_cat.astype(float), scaled_train_num])
X_test_no_emb = np.hstack([encoded_test_cat.astype(float), scaled_test_num])

model.fit(X_train_no_emb, y_train)
y_pred = model.predict(X_test_no_emb)

# A3: No origin, change, urgency
cat_cols = ['Type', 'Origin', 'Change', 'Urgency']
drop_cols = ['Origin', 'Change', 'Urgency']

keep_idx = [i for i, c in enumerate(cat_cols) if c not in drop_cols]
encoded_train_cat_reduced = encoded_train_cat[:, keep_idx]
encoded_test_cat_reduced = encoded_test_cat[:, keep_idx]

X_train_no_main_cat = np.hstack([embeddings_train_reduced*0.5, encoded_train_cat_reduced.astype(float), scaled_train_num])
X_test_no_main_cat = np.hstack([embeddings_test_reduced*0.5, encoded_test_cat_reduced.astype(float), scaled_test_num])

model.fit(X_train_no_main_cat, y_train)
y_pred = model.predict(X_test_no_main_cat)

#####################
'''

model.fit(X_train_full, y_train)
y_pred = model.predict(X_test_full)

decoded_preds = target_encoder.inverse_transform(y_pred.reshape(-1, 1)).flatten()
counts = pd.Series(decoded_preds).value_counts()
print("\nPredictions for each class of priority:")
print(counts)

# Evaluation
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_encoder.categories_[0], yticklabels=target_encoder.categories_[0])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.categories_[0]))

# Cross validation with accuracy score
accuracy_scores = cross_val_score(model, X_train_full, y_train, cv=5, scoring='accuracy')
print("Cross-validated Accuracy scores: [" + ", ".join(f"{score:.2f}" for score in accuracy_scores) + "]")
print(f"Mean Accuracy: {accuracy_scores.mean():.2f}")

# Macro F1-score
f1_macro_score = f1_score(y_test, y_pred, average='macro')
print(f"Mean Macro F1: {f1_macro_score:.2f}")

# Accuracy score
accuracy_score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_score:.2f}")


#######################################################################################################################

# Features importances plot
feature_importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()

embedding_importance = feature_importances[embedding_cols_reduced].sum()
other_features_importances = feature_importances.drop(embedding_cols_reduced)

fi_plot_df = pd.DataFrame({
    'feature': ['Embeddings'] + list(other_features_importances.index),
    'importance': [embedding_importance] + list(other_features_importances.values)
})

fi_plot_df = fi_plot_df.sort_values('importance', ascending=True)

colors = []
for f in fi_plot_df['feature']:
    if f == 'Embeddings':
        colors.append('skyblue')
    elif f in ['Type', 'Change', 'Urgency', 'Origin']:
        colors.append('lightgreen')
    else:
        colors.append('orange')

plt.figure(figsize=(10,6))
plt.barh(fi_plot_df['feature'], fi_plot_df['importance'], color=colors)
plt.xscale('log')
plt.xlabel('Estimated Feature Importance')
plt.title('Features Importances')
plt.tight_layout()
plt.show()

#######################################################################################################################

joblib.dump(model, "../models/classifier.pkl")
joblib.dump(pca_emb, "../models/pca.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(target_encoder, "../models/encoder.pkl")


