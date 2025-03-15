import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

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
train_data = pd.read_csv('../data/pred_data/train_dataset.csv')
test_data = pd.read_csv('../data/pred_data/test_dataset.csv')

# Cross-validation setup
for fold, (train_idx, val_idx) in enumerate(cv.split(train_data)):
    # Split the data into training and validation sets for this fold
    train_data_fold = train_data.iloc[train_idx]
    val_data_fold = train_data.iloc[val_idx]

    # Prepare the features for both training and validation data using the prepare_features function
    X_train_fold = train_data_fold.drop(['id', 'Label']).values
    # prepare_features(train_data_fold, dc.X_train_bow_q2, dc.X_train_bow_q4, dc.X_train_bow_q5, dc.X_train_bow_q6)
    X_valid_fold = val_data_fold.drop(['id', 'Label']).values 
    # prepare_features(val_data_fold, dc.X_valid_bow_q2, dc.X_valid_bow_q4, dc.X_valid_bow_q5, dc.X_valid_bow_q6)

    # TRAIN MODEL

    # ACCURACY
