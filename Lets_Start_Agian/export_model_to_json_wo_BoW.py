import os
import json
import joblib
import numpy as np

# Step 1: Paths
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'mlp_model_reduced_wo_BoW.pkl')
scaler_path = os.path.join(base_dir, 'scaler_wo_BoW.pkl')
encoder_path = os.path.join(base_dir, 'label_encoder_wo_BoW.pkl')
pca_path = os.path.join(base_dir, 'pca_wo_BoW.pkl')  # optional
feature_idx_path = os.path.join(base_dir, 'top_feature_indices_wo_BoW.pkl')  # optional

# Step 2: Load components
mlp = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

# Export MLP weights and biases
weights = [w.tolist() for w in mlp.coefs_]
biases = [b.tolist() for b in mlp.intercepts_]

# Export scaler
scaler_data = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist()
}

# Export label encoder
labels = label_encoder.classes_.tolist()

# Step 3: Export reducer info
if os.path.exists(feature_idx_path):
    top_indices = joblib.load(feature_idx_path)
    reducer = {
        "type": "feature_selection",
        "top_indices": top_indices.tolist()
    }
elif os.path.exists(pca_path):
    pca = joblib.load(pca_path)
    reducer = {
        "type": "pca",
        "components": pca.components_.tolist(),
        "explained_variance": pca.explained_variance_ratio_.tolist()
    }
else:
    reducer = {
        "type": "none"
    }

# Step 4: Combine everything
exported_model = {
    "weights": weights,
    "biases": biases,
    "scaler": scaler_data,
    "label_encoder": labels,
    "reducer": reducer,
    "activation": mlp.activation,
    "hidden_layer_sizes": mlp.hidden_layer_sizes
}

# Step 5: Save to JSON
output_path = os.path.join(base_dir, 'mlp_model_export_wo_BoW.json')
with open(output_path, 'w') as f:
    json.dump(exported_model, f, indent=2)

print(f"✅ Model exported to {output_path}")
