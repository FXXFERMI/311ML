import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Load datasets
train_main = pd.read_csv('./dataset/train_dataset1.csv')  # Main dataset
train_bow_q5 = pd.read_csv('./dataset/vocabQ5.csv')  # BoW representation for Q5
train_bow_q6 = pd.read_csv('./dataset/vocabQ6.csv')  # BoW representation for Q6

# Merge datasets on 'ID'
train_data = train_main.merge(train_bow_q5, on='ID', how='left')\
                       .merge(train_bow_q6, on='ID', how='left')


# === Constants ===
feature_columns = [
    'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)',
    'Q2: How many ingredients would you expect this food item to contain?',
    'Q4: How much would you expect to pay for one serving of this food item?'
]

q3_choices = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
q7_choices = ['Parents', 'Siblings', 'Friends', 'Teachers', 'Strangers']
q8_choices = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)', 'I will have some of this food item with my hot sauce']

q5 = "Q5: What movie do you think of when thinking of this food item?"
q6 = "Q6: What drink would you pair with this food item?"

vocabQ5 = pd.read_csv('./dataset/vocabQ5.csv').drop(columns=['ID']).columns.tolist()
vocabQ6 = pd.read_csv('./dataset/vocabQ6.csv').drop(columns=['ID']).columns.tolist()

median_values = train_main[feature_columns].median()
mean_values = train_main[feature_columns].mean()
std_values = train_main[feature_columns].std()

label_mapping = sorted(train_main["Label"].unique())

# === Helper functions ===
def extract_numbers(text):
    if pd.isna(text):
        return np.nan
    text = str(text).lower()
    numbers = []
    for word in text.split():
        if '-' in word or 'to' in word:
            values = [int(x) for x in word.replace('to', '-').split('-') if x.isdigit()]
            if len(values) == 2:
                numbers.append(np.mean(values))
        elif word.isdigit():
            numbers.append(int(word))
    return np.nan if not numbers else np.mean(numbers)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def preprocess_test_data(test_df):
    for col in feature_columns:
        if col in ['Q2: How many ingredients would you expect this food item to contain?',
                   'Q4: How much would you expect to pay for one serving of this food item?']:
            test_df[col] = test_df[col].astype(str).apply(extract_numbers)
        test_df[col] = test_df[col].fillna(median_values[col])
        test_df[col] = (test_df[col] - mean_values[col]) / std_values[col]

    for c in q3_choices + q7_choices + q8_choices:
        test_df[c] = 0

    q3_clean = test_df["Q3: In what setting would you expect this food to be served? Please check all that apply"].fillna('').str.split(',')
    for i, choices in q3_clean.items():
        for c in choices:
            c = c.strip()
            if c in q3_choices:
                test_df.at[i, c] = 1

    q7_clean = test_df["Q7: When you think about this food item, who does it remind you of?"].fillna('').str.split(',')
    for i, choices in q7_clean.items():
        for c in choices:
            c = c.strip()
            if c in q7_choices:
                test_df.at[i, c] = 1

    q8_clean = test_df["Q8: How much hot sauce would you add to this food item?"].fillna('')
    for i, choice in q8_clean.items():
        if choice in q8_choices:
            test_df.at[i, choice] = 1
        else:
            test_df.at[i, 'None'] = 1

    test_df[q5] = test_df[q5].fillna("unknown").str.lower().str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
    bow_q5 = pd.DataFrame(0, index=test_df.index, columns=vocabQ5)
    for i, row in test_df[q5].items():
        for word in row.split():
            if word in vocabQ5:
                bow_q5.at[i, word] += 1

    test_df[q6] = test_df[q6].fillna("unknown").str.lower().str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
    bow_q6 = pd.DataFrame(0, index=test_df.index, columns=vocabQ6)
    for i, row in test_df[q6].items():
        for word in row.split():
            if word in vocabQ6:
                bow_q6.at[i, word] += 1

    X = np.hstack((
        test_df[feature_columns + q3_choices + q7_choices + q8_choices].values,
        bow_q5.values,
        bow_q6.values
    ))
    return X, test_df['Label'].values if 'Label' in test_df.columns else None

# Define K-Fold cross-validator
cv = KFold(n_splits=5, random_state=18, shuffle=True)

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
                          alpha=0.001, max_iter=200, random_state=18, verbose=True)

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
        best_loss_curve = model.loss_curve_

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
pd.DataFrame(best_weights['W1']).to_csv(f'{weights_dir}best_W1.csv', index=False, header=False)
pd.DataFrame(best_weights['b1']).to_csv(f'{weights_dir}best_b1.csv', index=False, header=False)
pd.DataFrame(best_weights['W2']).to_csv(f'{weights_dir}best_W2.csv', index=False, header=False)
pd.DataFrame(best_weights['b2']).to_csv(f'{weights_dir}best_b2.csv', index=False, header=False)
pd.DataFrame(best_weights['W3']).to_csv(f'{weights_dir}best_W3.csv', index=False, header=False)
pd.DataFrame(best_weights['b3']).to_csv(f'{weights_dir}best_b3.csv', index=False, header=False)

print("\n✅ Best model weights stored as CSV files!")


# === Evaluate model on test.csv ===
test_path = './dataset/test.csv'

if os.path.exists(test_path):
    test_df = pd.read_csv(test_path)

    if 'Label' not in test_df.columns:
        print("\n⚠️  'Label' column not found in test.csv — can't calculate test accuracy.")
    else:
        X_test, y_test = preprocess_test_data(test_df)
        X_test = scaler.transform(X_test)

        # Manually forward pass using best weights
        A1 = relu(X_test @ best_weights['W1'] + best_weights['b1'])
        A2 = relu(A1 @ best_weights['W2'] + best_weights['b2'])
        probs = softmax(A2 @ best_weights['W3'] + best_weights['b3'])

        y_pred = np.argmax(probs, axis=1)
        label_to_index = {label: idx for idx, label in enumerate(label_mapping)}
        y_test_idx = [label_to_index[y] for y in y_test]

        test_acc = accuracy_score(y_test_idx, y_pred)
        print(f"\n✅ MLP Test Accuracy: {test_acc:.4f}")
else:
    print("\n⚠️ test.csv not found — skipping test accuracy evaluation.")


########### Tuning hyperParameters #########
plt.figure(figsize=(8, 5))
plt.plot(best_loss_curve, label='Training Loss')
plt.title(f'Best Fold {best_fold} Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()