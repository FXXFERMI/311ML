import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Define the K-Fold cross-validator (5-fold)
cv = KFold(n_splits=5, random_state=18, shuffle=True)

# Load datasets
train_main = pd.read_csv('./dataset/train_dataset.csv')  # Main dataset
train_bow_q5 = pd.read_csv('./dataset/vocabQ5.csv')  # BoW representation for Q5
train_bow_q6 = pd.read_csv('./dataset/vocabQ6.csv')  # BoW representation for Q6

# Merge datasets on 'ID'
train_data = train_main.merge(train_bow_q5, on='ID', how='left')\
                       .merge(train_bow_q6, on='ID', how='left')

# Store accuracy results
accuracy_scores = []

# Cross-validation setup
for fold, (train_idx, val_idx) in enumerate(cv.split(train_data)):
    print(f"Training fold {fold + 1}...")

    # Split the data into training and validation sets
    train_data_fold = train_data.iloc[train_idx]
    val_data_fold = train_data.iloc[val_idx]

    # Extract features and labels
    X_train_fold = train_data_fold.drop(['ID', 'Label'], axis=1).values
    y_train_fold = train_data_fold['Label'].values
    X_valid_fold = val_data_fold.drop(['ID', 'Label'], axis=1).values
    y_valid_fold = val_data_fold['Label'].values

    # Initialize Decision Tree model
    model = DecisionTreeClassifier(max_depth=20, random_state=18)

    # Train the model
    model.fit(X_train_fold, y_train_fold)

    # Predict on validation set
    y_pred = model.predict(X_valid_fold)

    # Evaluate accuracy
    accuracy = accuracy_score(y_valid_fold, y_pred)
    accuracy_scores.append(accuracy)

    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

# Compute average accuracy across all folds
average_accuracy = np.mean(accuracy_scores)
print(f"\nAverage Cross-Validation Accuracy: {average_accuracy:.4f}")
