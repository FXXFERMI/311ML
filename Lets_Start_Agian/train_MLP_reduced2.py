import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from data_clean import process_clean_data

# === CONFIG ===
USE_PCA = False         # Set to True for PCA, False for feature selection
N_FEATURES = 100        # Used only if USE_PCA is False

# === Step 1: Load cleaned data ===
base_dir = os.path.dirname(__file__)
clean_data_path = os.path.join(base_dir, 'cleaned_train_dataset.csv')
df = pd.read_csv(clean_data_path)

# === Step 2: Prepare X and y ===
y = df['Label']
X = df.drop(columns=['Label'])

# === Step 3: Encode labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Step 4: Standardize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 5: Dimensionality Reduction ===
if USE_PCA:
    print("🔧 Applying PCA (retain 95% variance)...")
    pca = PCA(n_components=0.95, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"PCA reduced from {X_scaled.shape[1]} → {X_reduced.shape[1]} dimensions")
    joblib.dump(pca, os.path.join(base_dir, 'pca.pkl'))
else:
    print(f"🔧 Selecting top {N_FEATURES} features by RandomForest importance...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_scaled, y_encoded)
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-N_FEATURES:]
    X_reduced = X_scaled[:, top_indices]
    joblib.dump(top_indices, os.path.join(base_dir, 'top_feature_indices.pkl'))

# === Step 6: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_encoded, test_size=0.2, random_state=42)

# === Step 7: Train MLP Model ===
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='tanh',
    alpha=0.01,
    learning_rate_init=0.01,
    max_iter=500,
    random_state=42
)
mlp.fit(X_train, y_train)

# === Step 8: Evaluate ===
y_pred = mlp.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === Step 9: Save model and preprocessors ===
joblib.dump(mlp, os.path.join(base_dir, 'mlp_model_reduced.pkl'))
joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))
joblib.dump(le, os.path.join(base_dir, 'label_encoder.pkl'))

print("✅ Reduced model, scaler, and encoder saved successfully.")
