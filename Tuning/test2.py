import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# === Load and split the original dataset ===
data_path = './dataset/train_dataset.csv'
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

# === Grid Search: alpha & hidden_layer_sizes ===
alphas = [0.0001, 0.001, 0.01]
hidden_layers = [(32,), (64,), (64, 32)]
results = []

for alpha in alphas:
    for hsize in hidden_layers:
        print(f"Training with alpha={alpha}, hidden_layer_sizes={hsize}")
        model = MLPClassifier(hidden_layer_sizes=hsize, alpha=alpha, max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        results.append({"alpha": alpha, "hidden_size": str(hsize), "val_acc": acc})

# === Heatmap Visualization ===
df_results = pd.DataFrame(results)
heatmap_data = df_results.pivot(index="hidden_size", columns="alpha", values="val_acc")
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title("Validation Accuracy Heatmap (alpha vs hidden_layer_sizes)")
plt.xlabel("Alpha")
plt.ylabel("Hidden Layer Sizes")
plt.tight_layout()
plt.show()

# === Pick best model ===
best_row = df_results.loc[df_results['val_acc'].idxmax()]
best_alpha = best_row['alpha']
best_hidden_size = eval(best_row['hidden_size'])

# === Retrain with best hyperparameters ===
best_model = MLPClassifier(hidden_layer_sizes=best_hidden_size, alpha=best_alpha, max_iter=500, random_state=42, verbose=True)
best_model.fit(X_train, y_train)

# === Save weights dynamically ===
best_weights = {}
for i in range(len(best_model.coefs_)):
    best_weights[f'W{i+1}'] = best_model.coefs_[i]
    best_weights[f'b{i+1}'] = best_model.intercepts_[i]

weights_dir = './best_model_weights/'
os.makedirs(weights_dir, exist_ok=True)

for name, value in best_weights.items():
    pd.DataFrame(value).to_csv(f'{weights_dir}best_{name}.csv', index=False, header=False)

print("\n✅ Best model weights stored as CSV files!")

# === Final test accuracy with dynamic forward pass ===
def relu(x):
    return np.maximum(0, x)

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

# === Loss curve ===
plt.figure(figsize=(8, 5))
plt.plot(best_model.loss_curve_, label='Training Loss')
plt.title('MLP Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()