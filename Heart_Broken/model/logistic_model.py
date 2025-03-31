import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define the K-Fold cross-validator (5-fold in this case)
cv = KFold(n_splits=5, random_state=18, shuffle=True)

# def prepare_features(data, *bow_features_list):
    # features_df = data.drop(['id', 'Label']) 
                             # 'Q2: How many ingredients would you expect this food item to contain?', 'Q4: How much would you expect to pay for one serving of this food item?', 'Q5: What movie do you think of when thinking of this food item?', 'Q6: What drink would you pair with this food item?'], axis=1)
    # bow_dfs = [pd.DataFrame(bow_features, index=features_df.index) for bow_features in bow_features_list]
    # combined_df = pd.concat([features_df] + bow_dfs, axis=1)
    # return combined_df.values
#     return data.drop(['id', 'Label']) 

# Load datasets
train_main = pd.read_csv('./dataset/train_dataset.csv')# Main dataset
train_bow_q5 = pd.read_csv('./dataset/vocabQ5.csv')  # BoW representation for Q5
train_bow_q6 = pd.read_csv('./dataset/vocabQ6.csv')  # BoW representation for Q6

test_data = pd.read_csv('./dataset/test.csv')

# Ensure they have a common column (like 'id') for merging
train_data = train_main.merge(train_bow_q5, on='ID', how='left')\
                       .merge(train_bow_q6, on='ID', how='left')



# Store accuracy results
accuracy_scores = []

# # Cross-validation setup
# for fold, (train_idx, val_idx) in enumerate(cv.split(train_data)):
#     # Split the data into training and validation sets for this fold
#     train_data_fold = train_data.iloc[train_idx]
#     val_data_fold = train_data.iloc[val_idx]

#     # Prepare the features for both training and validation data using the prepare_features function
#     X_train_fold = train_data_fold.drop(['ID', 'Label']).values
#     # prepare_features(train_data_fold, dc.X_train_bow_q2, dc.X_train_bow_q4, dc.X_train_bow_q5, dc.X_train_bow_q6)
#     X_valid_fold = val_data_fold.drop(['ID', 'Label']).values 
#     # prepare_features(val_data_fold, dc.X_valid_bow_q2, dc.X_valid_bow_q4, dc.X_valid_bow_q5, dc.X_valid_bow_q6)

#     # TRAIN MODEL

#     # ACCURACY

# Cross-validation setup
for fold, (train_idx, val_idx) in enumerate(cv.split(train_data)):
    print(f"Training fold {fold + 1}...")

    # Split the data into training and validation sets for this fold
    train_data_fold = train_data.iloc[train_idx]
    val_data_fold = train_data.iloc[val_idx]

    # Extract features and labels
    X_train_fold = train_data_fold.drop(['ID', 'Label'], axis=1).values
    y_train_fold = train_data_fold['Label'].values
    X_valid_fold = val_data_fold.drop(['ID', 'Label'], axis=1).values
    y_valid_fold = val_data_fold['Label'].values

    # Initialize Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=18)

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
