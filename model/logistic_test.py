import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# K-Fold setup
cv = KFold(n_splits=5, shuffle=True, random_state=18)

# === Global Constants ===

# Columns to normalize
feature_columns = [
    'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)',
    'Q2: How many ingredients would you expect this food item to contain?',
    'Q4: How much would you expect to pay for one serving of this food item?'
]

# Columns to one-hot encode
q3_choices = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
q7_choices = ['Parents', 'Siblings', 'Friends', 'Teachers', 'Strangers']
q8_choices = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)', 'I will have some of this food item with my hot sauce']

# Columns for BoW
q5 = "Q5: What movie do you think of when thinking of this food item?"
q6 = "Q6: What drink would you pair with this food item?"

# Vocabularies for BoW
vocabQ5 = pd.read_csv('./dataset/vocabQ5.csv').drop(columns=['ID']).columns.tolist()
vocabQ6 = pd.read_csv('./dataset/vocabQ6.csv').drop(columns=['ID']).columns.tolist()

# Load training data for stats
train_data = pd.read_csv('./dataset/train_dataset.csv')
median_values = train_data[feature_columns].median()
mean_values = train_data[feature_columns].mean()
std_values = train_data[feature_columns].std()


# Load and merge data
train_main = pd.read_csv('./dataset/train_dataset.csv')
train_bow_q5 = pd.read_csv('./dataset/vocabQ5.csv')
train_bow_q6 = pd.read_csv('./dataset/vocabQ6.csv')
train_data = train_main.merge(train_bow_q5, on='ID', how='left')\
                       .merge(train_bow_q6, on='ID', how='left')

# Standardization setup
scaler = StandardScaler()

# Best model tracking
best_model = None
best_accuracy = 0
best_fold = -1
best_weights = {}

accuracy_scores = []

# Cross-validation training
for fold, (train_idx, val_idx) in enumerate(cv.split(train_data)):
    print(f"Training fold {fold + 1}...")

    train_fold = train_data.iloc[train_idx]
    val_fold = train_data.iloc[val_idx]

    X_train = train_fold.drop(['ID', 'Label'], axis=1).values
    y_train = train_fold['Label'].values
    X_val = val_fold.drop(['ID', 'Label'], axis=1).values
    y_val = val_fold['Label'].values

    # Scale features
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=18)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracy_scores.append(accuracy)

    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

    # Save best model weights
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy
        best_fold = fold + 1
        best_weights = {
            'W': best_model.coef_,  # shape: (n_classes, n_features)
            'b': best_model.intercept_  # shape: (n_classes,)
        }

# Final stats
print("\nAccuracy Scores:", accuracy_scores)
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Best Fold: {best_fold} with Accuracy: {best_accuracy:.4f}")

# Save weights to file
weights_dir = './best_logreg_weights/'
os.makedirs(weights_dir, exist_ok=True)

pd.DataFrame(best_weights['W']).to_csv(os.path.join(weights_dir, 'W.csv'), index=False, header=False)
pd.DataFrame(best_weights['b']).to_csv(os.path.join(weights_dir, 'b.csv'), index=False, header=False)

print("\n✅ Best logistic regression model weights saved!")

# === Evaluate on test set ===
# === Evaluate best model on test.csv ===
# === Helper: extract number ranges like "2 to 3" or "5-7" ===
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


# Copying preprocessing from predict_all()
def preprocess_data(df):
    # Clean numeric fields (Q1, Q2, Q4)
    for col in feature_columns:
        if col in ['Q2: How many ingredients would you expect this food item to contain?', 
                   'Q4: How much would you expect to pay for one serving of this food item?']:
            df[col] = df[col].astype(str).apply(extract_numbers)
        df[col] = df[col].fillna(median_values[col])
        df[col] = (df[col] - mean_values[col]) / std_values[col]

    # One-hot encoding
    for c in q3_choices + q7_choices + q8_choices:
        df[c] = 0

    q3_clean = df["Q3: In what setting would you expect this food to be served? Please check all that apply"].fillna('').str.split(',')
    for i, choices in q3_clean.items():
        for c in choices:
            c = c.strip()
            if c in q3_choices:
                df.at[i, c] = 1

    q7_clean = df["Q7: When you think about this food item, who does it remind you of?"].fillna('').str.split(',')
    for i, choices in q7_clean.items():
        for c in choices:
            c = c.strip()
            if c in q7_choices:
                df.at[i, c] = 1

    q8_clean = df["Q8: How much hot sauce would you add to this food item?"].fillna('')
    for i, choice in q8_clean.items():
        if choice in q8_choices:
            df.at[i, choice] = 1
        else:
            df.at[i, 'None'] = 1

    # BoW Q5
    df[q5] = df[q5].fillna("unknown").str.lower().str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
    bow_q5 = pd.DataFrame(0, index=df.index, columns=vocabQ5)
    for i, row in df[q5].items():
        for word in row.split():
            if word in vocabQ5:
                bow_q5.at[i, word] += 1

    # BoW Q6
    df[q6] = df[q6].fillna("unknown").str.lower().str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
    bow_q6 = pd.DataFrame(0, index=df.index, columns=vocabQ6)
    for i, row in df[q6].items():
        for word in row.split():
            if word in vocabQ6:
                bow_q6.at[i, word] += 1

    # Combine all features
    X = np.hstack((
        df[feature_columns + q3_choices + q7_choices + q8_choices].values,
        bow_q5.values,
        bow_q6.values
    ))
    return X


# === Constants from preprocessing ===
q5 = "Q5: What movie do you think of when thinking of this food item?"
q6 = "Q6: What drink would you pair with this food item?"
q3_choices = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
q7_choices = ['Parents', 'Siblings', 'Friends', 'Teachers', 'Strangers']
q8_choices = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)', 'I will have some of this food item with my hot sauce']

# === Load test set
test_path = './dataset/test.csv'
if os.path.exists(test_path):
    test_df = pd.read_csv(test_path)

    if 'Label' not in test_df.columns:
        print("\n⚠️  'Label' column not found in test.csv — can't calculate test accuracy.")
    else:
        # Preprocess test data
        X_test = preprocess_data(test_df)
        y_test = test_df['Label'].values

        # Apply scaler
        X_test = scaler.transform(X_test)

        # Predict using best logistic model
        y_pred_test = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)

        print(f"\n✅ Logistic Regression Test Accuracy: {test_acc:.4f}")
else:
    print("\n⚠️ test.csv not found — skipping test accuracy evaluation.")
