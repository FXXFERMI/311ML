# === MLP Training with 80/15/5 Split ===

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# === Load and split the original dataset ===
data_path = './dataset/train_dataset1.csv'
data = pd.read_csv(data_path)

# Split into train (80%) and temp (20%)
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Label'])

# Split temp into val (15%) and test (5%)
val_data, test_data = train_test_split(temp_data, test_size=0.25, random_state=42, stratify=temp_data['Label'])

print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")

# === Feature columns ===
feature_columns = [
    'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)',
    'Q2: How many ingredients would you expect this food item to contain?',
    'Q4: How much would you expect to pay for one serving of this food item?'
]

q3_choices = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
q7_choices = ['Parents', 'Siblings', 'Friends', 'Teachers', 'Strangers']
q8_choices = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)', 'I will have some of this food item with my hot sauce']

all_features = feature_columns + q3_choices + q7_choices + q8_choices
label_mapping = sorted(data['Label'].unique())

# === Extract features and labels ===
def extract_X_y(df):
    X = df[all_features].values
    y = df['Label'].values
    return X, y

X_train, y_train = extract_X_y(train_data)
X_val, y_val = extract_X_y(val_data)
X_test, y_test = extract_X_y(test_data)

# === Standardize features ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# === Train the model ===
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                      alpha=0.001, max_iter=500, random_state=42, verbose=True)

model.fit(X_train, y_train)

# === Validation accuracy ===
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"\n✅ Validation Accuracy: {val_acc:.4f}")

# === Save best model weights ===
best_weights = {
    'W1': model.coefs_[0], 'b1': model.intercepts_[0],
    'W2': model.coefs_[1], 'b2': model.intercepts_[1],
    'W3': model.coefs_[2], 'b3': model.intercepts_[2],
}

weights_dir = './best_model_weights/'
os.makedirs(weights_dir, exist_ok=True)

pd.DataFrame(best_weights['W1']).to_csv(f'{weights_dir}best_W1.csv', index=False, header=False)
pd.DataFrame(best_weights['b1']).to_csv(f'{weights_dir}best_b1.csv', index=False, header=False)
pd.DataFrame(best_weights['W2']).to_csv(f'{weights_dir}best_W2.csv', index=False, header=False)
pd.DataFrame(best_weights['b2']).to_csv(f'{weights_dir}best_b2.csv', index=False, header=False)
pd.DataFrame(best_weights['W3']).to_csv(f'{weights_dir}best_W3.csv', index=False, header=False)
pd.DataFrame(best_weights['b3']).to_csv(f'{weights_dir}best_b3.csv', index=False, header=False)

print("\n✅ Best model weights stored as CSV files!")

# === Evaluate on test set ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

A1 = relu(X_test @ best_weights['W1'] + best_weights['b1'])
A2 = relu(A1 @ best_weights['W2'] + best_weights['b2'])
probs = softmax(A2 @ best_weights['W3'] + best_weights['b3'])

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