import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# === Load datasets ===
train_main = pd.read_csv('./dataset/train_dataset.csv')
train_bow_q2 = pd.read_csv('./dataset/vocabQ2.csv')
train_bow_q4 = pd.read_csv('./dataset/vocabQ4.csv')
train_bow_q5 = pd.read_csv('./dataset/vocabQ5.csv')
train_bow_q6 = pd.read_csv('./dataset/vocabQ6.csv')

# === Merge all BoW datasets safely ===
def merge_and_add(df1, df2):
    common_cols = set(df1.columns).intersection(df2.columns) - {'ID'}
    for col in common_cols:
        df1[col] = df1[col].fillna(0) + df2[col].fillna(0)
    df2 = df2.drop(columns=common_cols, errors='ignore')
    return df1.merge(df2, on='ID', how='left')

merged_bow = merge_and_add(train_bow_q2.copy(), train_bow_q4)
merged_bow = merge_and_add(merged_bow, train_bow_q5)
merged_bow = merge_and_add(merged_bow, train_bow_q6)
merged_bow = merged_bow.fillna(0)

# === Merge BoW with main dataset ===
columns_to_drop = [
    'Q2: How many ingredients would you expect this food item to contain?',
    'Q4: How much would you expect to pay for one serving of this food item?',
    'Q5: What movie do you think of when thinking of this food item?',
    'Q6: What drink would you pair with this food item?'
]

train_main = train_main.drop(columns=columns_to_drop, errors="ignore")
train_data = train_main.merge(merged_bow, on='ID', how='left')

# === Feature columns ===
feature_columns = [
    'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)',
    'Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack',
    'Parents', 'Siblings', 'Friends', 'Teachers', 'Strangers',
    'None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)', 'I will have some of this food item with my hot sauce'
] + list(merged_bow.columns.difference(['ID']))

label_mapping = sorted(train_data['Label'].unique())

# === Extract features and labels ===
def extract_X_y(df):
    X = df[feature_columns].values
    y = df['Label'].values
    return X, y

# === Split into train/val/test ===
train_set, temp_set = train_test_split(train_data, test_size=0.2, stratify=train_data['Label'], random_state=42)
val_set, test_set = train_test_split(temp_set, test_size=0.25, stratify=temp_set['Label'], random_state=42)

print(f"Train size: {len(train_set)}, Val size: {len(val_set)}, Test size: {len(test_set)}")

X_train, y_train = extract_X_y(train_set)
X_val, y_val = extract_X_y(val_set)
X_test, y_test = extract_X_y(test_set)

# === Standardization ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# === Train MLP ===
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                      alpha=0.001, max_iter=500, random_state=42, verbose=True)
model.fit(X_train, y_train)

# === Validation accuracy ===
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"\n✅ Validation Accuracy: {val_acc:.4f}")

# === Save best model weights ===
best_weights = {f"W{i+1}": w for i, w in enumerate(model.coefs_)}
best_weights.update({f"b{i+1}": b for i, b in enumerate(model.intercepts_)})

weights_dir = './best_model_weights/'
os.makedirs(weights_dir, exist_ok=True)

for name, value in best_weights.items():
    pd.DataFrame(value).to_csv(f'{weights_dir}best_{name}.csv', index=False, header=False)

print("\n✅ Best model weights stored as CSV files!")

# === Test accuracy ===
def relu(x): return np.maximum(0, x)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_pass(X, weights):
    A = X
    num_layers = len([k for k in weights if k.startswith('W')])
    for i in range(1, num_layers):
        A = relu(A @ weights[f'W{i}'] + weights[f'b{i}'])
    logits = A @ weights[f'W{num_layers}'] + weights[f'b{num_layers}']
    return softmax(logits)

probs = forward_pass(X_test, best_weights)
y_pred = np.argmax(probs, axis=1)
label_to_index = {label: idx for idx, label in enumerate(label_mapping)}
y_test_idx = [label_to_index[y] for y in y_test]
test_acc = accuracy_score(y_test_idx, y_pred)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# === Plot training loss ===
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_, label='Training Loss')
plt.title('MLP Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()