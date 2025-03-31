import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Define K-Fold cross-validator
cv = KFold(n_splits=5, random_state=18, shuffle=True)

# Load datasets
train_main = pd.read_csv('./dataset/train_dataset.csv')  # Main dataset
train_bow_q2 = pd.read_csv('./dataset/vocabQ2.csv')  # BoW representation for Q2
train_bow_q4 = pd.read_csv('./dataset/vocabQ4.csv')  # BoW representation for Q4
train_bow_q5 = pd.read_csv('./dataset/vocabQ5.csv')  # BoW representation for Q5
train_bow_q6 = pd.read_csv('./dataset/vocabQ6.csv')  # BoW representation for Q6

# Start with the first dataset
merged_bow = train_bow_q2.copy()

# Define a function to merge and add overlapping columns
def merge_and_add(df1, df2):
    # Identify common columns (excluding ID)
    common_cols = set(df1.columns).intersection(set(df2.columns)) - {"ID"}
    
    # Add overlapping column values
    for col in common_cols:
        df1[col] = df1[col].fillna(0) + df2[col].fillna(0)
    
    # Drop overlapping columns from df2 (since they've been added to df1)
    df2 = df2.drop(columns=common_cols, errors="ignore")
    
    # Merge the updated df1 and df2
    return df1.merge(df2, on="ID", how="left")

# Merge all BoW datasets using the function
merged_bow = merge_and_add(merged_bow, train_bow_q4)
merged_bow = merge_and_add(merged_bow, train_bow_q5)
merged_bow = merge_and_add(merged_bow, train_bow_q6)

# Fill NaN values with 0 after merging
merged_bow = merged_bow.fillna(0)

# Merge datasets on 'ID'
train_data = train_main.merge(merged_bow, on='ID', how='left')
                       

# Standardize features (MLP requires scaling)
scaler = StandardScaler()

# Store best model information
best_model = None
best_accuracy = 0
best_fold = -1
best_weights = {}  # Dictionary to store the best model's weights

# Store accuracy results
accuracy_scores = []

# Train models using K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(cv.split(train_data)):
    print(f"Training fold {fold + 1}...")

    # Split data into train and validation sets
    train_data_fold = train_data.iloc[train_idx]
    val_data_fold = train_data.iloc[val_idx]

    # Extract features and labels
    X_train_fold = train_data_fold.drop(['ID', 'Label'], axis=1).values
    y_train_fold = train_data_fold['Label'].values
    X_valid_fold = val_data_fold.drop(['ID', 'Label'], axis=1).values
    y_valid_fold = val_data_fold['Label'].values

    # Scale features
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_valid_fold = scaler.transform(X_valid_fold)

    # Initialize MLP model
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                          alpha=0.001, max_iter=500, random_state=18)

    # Train the model
    model.fit(X_train_fold, y_train_fold)

    # Predict on validation set
    y_pred = model.predict(X_valid_fold)

    # Evaluate accuracy
    accuracy = accuracy_score(y_valid_fold, y_pred)
    accuracy_scores.append(accuracy)

    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

    # Store the model if it has the best accuracy
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy
        best_fold = fold + 1

        # Save the best model's weights
        best_weights = {
            'W1': best_model.coefs_[0], 'b1': best_model.intercepts_[0],
            'W2': best_model.coefs_[1], 'b2': best_model.intercepts_[1],
            'W3': best_model.coefs_[2], 'b3': best_model.intercepts_[2],
        }

# Print final accuracy statistics
print("\nCollected Accuracy Scores:", accuracy_scores)

# Compute average accuracy **only if scores exist**
if len(accuracy_scores) > 0:
    average_accuracy = np.mean(accuracy_scores)
else:
    average_accuracy = 0  # Avoid NaN errors

print(f"\nFinal Average Cross-Validation Accuracy: {average_accuracy:.4f}")
print(f"\nBest model found at Fold {best_fold} with Accuracy: {best_accuracy:.4f}")

# ✅ Ensure the directory exists before saving model weights
weights_dir = './best_model_weights/'

if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

# ✅ Save best model weights to CSV files
pd.DataFrame(best_weights['W1']).to_csv(f'{weights_dir}best_W1_3.csv', index=False, header=False)
pd.DataFrame(best_weights['b1']).to_csv(f'{weights_dir}best_b1_3.csv', index=False, header=False)
pd.DataFrame(best_weights['W2']).to_csv(f'{weights_dir}best_W2_3.csv', index=False, header=False)
pd.DataFrame(best_weights['b2']).to_csv(f'{weights_dir}best_b2_3.csv', index=False, header=False)
pd.DataFrame(best_weights['W3']).to_csv(f'{weights_dir}best_W3_3.csv', index=False, header=False)
pd.DataFrame(best_weights['b3']).to_csv(f'{weights_dir}best_b3_3.csv', index=False, header=False)

print("\n✅ Best model weights stored as CSV files!")
