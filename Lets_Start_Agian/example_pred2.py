import os
import pandas as pd
import joblib
from data_clean import process_clean_data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Paths
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'mlp_model_reduced.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')
encoder_path = os.path.join(base_dir, 'label_encoder.pkl')
top_idx_path = os.path.join(base_dir, 'top_feature_indices.pkl')  # <-- Load feature indices
raw_test_path = os.path.join(base_dir, 'test_dataset.csv')

# Step 2: Load model + transformers
mlp_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)
top_indices = joblib.load(top_idx_path)

# Step 3: Load test data and extract true labels
raw_test_df = pd.read_csv(raw_test_path)
true_labels = None
if 'Label' in raw_test_df.columns:
    true_labels = raw_test_df['Label']

# Step 4: Clean the test data
cleaned_test_df = process_clean_data(raw_test_path)

# Step 5: Standardize and select top features
test_scaled = scaler.transform(cleaned_test_df)
# # Save sklearn's standardized version
# with open("test_scaled_sklearn.txt", "w") as f2:
#     for row in test_scaled:  # this is a NumPy array
#         line = ",".join(f"{val:.6f}" for val in row)
#         f2.write(line + "\n")
test_reduced = test_scaled[:, top_indices]
# # Save sklearn reduced
# with open("test_reduced_sklearn.txt", "w") as f:
#     for row in test_reduced:
#         f.write(",".join(f"{x:.6f}" for x in row) + "\n")


# Step 6: Predict
predictions = mlp_model.predict(test_reduced)
# for i, label in enumerate(predictions):
#     print(f"Sample {i+1}: {label}")
predicted_labels = label_encoder.inverse_transform(predictions)

# Step 7: Output
print("🔮 Predicted Food Labels:")
for i, label in enumerate(predicted_labels):
    print(f"Sample {i + 1}: {label}")

# Step 8: Evaluate (if true labels provided)
if true_labels is not None:
    true_encoded = label_encoder.transform(true_labels)
    acc = accuracy_score(true_encoded, predictions)
    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(true_encoded, predictions, target_names=label_encoder.classes_))

    # Confusion Matrix
    cm = confusion_matrix(true_encoded, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix (Test Set)")
    plt.show()
else:
    print("\nℹ️ No ground truth labels found in test dataset — skipping evaluation.")
