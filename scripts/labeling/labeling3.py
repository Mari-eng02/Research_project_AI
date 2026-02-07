import pandas as pd

# Datasets loading
df_train = pd.read_csv("../../dataset/train_labeled.csv")
df_test = pd.read_csv("../../dataset/test_labeled.csv")

# Concatenation of training and test set (temporary)
df = pd.concat([df_train,df_test], ignore_index=True).reset_index(drop=True)

# Heuristics to predict cost (in person-days) and time (in days)
def estimate_cost(row):
    cost = 1  # base cost in person-days

    # Based on Change type
    if row['Change'] == 'addition':
        cost += 2
    elif row['Change'] == 'deletion':
        cost += 1
    elif row['Change'] == 'modification':
        cost += 1.5

    # Based on Urgency
    if row['Urgency'] == 'soon':
        cost -= 1
    elif row['Urgency'] == 'later':
        cost += 2

    # Based on dependencies
    deps = int(row['N_dependencies'])
    cost += deps * 0.5

    return max(1, round(cost))  # Ensure cost ≥ 1

def estimate_time(row):
    time = estimate_cost(row) * 1.5  # time in calendar days

    # Adjust for urgency
    if row['Urgency'] == 'soon':
        time -= 1
    elif row['Urgency'] == 'later':
        time += 2

    return max(1, round(time))  # Ensure time ≥ 1


# Apply heuristics to the dataset
df['Cost'] = df.apply(estimate_cost, axis=1)
df['Time'] = df.apply(estimate_time, axis=1)

# Predictions analysis
print("Cost Estimations statistics:")
print(f"Min: {df['Cost'].min()}")
print(f"Max: {df['Cost'].max()}")
print(f"Mean: {df['Cost'].mean():.2f}")

print("\nTime Estimations statistics:")
print(f"Min: {df['Time'].min()}")
print(f"Max: {df['Time'].max()}")
print(f"Mean: {df['Time'].mean():.2f}")

# Saving data in the original datasets
df_train['Cost'] = df.loc[:len(df_train) - 1, 'Cost'].values
df_train['Time'] = df.loc[:len(df_train) - 1, 'Time'].values

df_test['Cost'] = df.loc[len(df_train):, 'Cost'].values
df_test['Time'] = df.loc[len(df_train):, 'Time'].values

df_train.to_csv("../../dataset/train_labeled.csv", index=False)
df_test.to_csv("../../dataset/test_labeled.csv", index=False)

print("\n✅ Dataset files saved with estimations")



