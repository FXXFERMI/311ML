import os
import pandas as pd
import joblib
from data_clean import process_clean_data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Set up paths
base_dir = os.path.dirname(__file__)
# model_path = os.path.join(base_dir, 'mlp_model.pkl')
# model_path = os.path.join(base_dir, 'mlp_model_tuned.pkl')
model_path = os.path.join(base_dir, 'mlp_model_reduced.pkl')
pca_path = os.path.join(base_dir, 'pca.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')
encoder_path = os.path.join(base_dir, 'label_encoder.pkl')
raw_test_path = os.path.join(base_dir, 'test_dataset.csv')

# Step 2: Load model, scaler, PCA, and label encoder
mlp_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)
label_encoder = joblib.load(encoder_path)

# Step 3: Load raw test data
raw_test_df = pd.read_csv(raw_test_path)
true_labels = None
if 'Label' in raw_test_df.columns:
    true_labels = raw_test_df['Label']
    raw_test_df = raw_test_df.drop(columns=['Label'])

# Step 4: Clean the raw test data (without Label)
cleaned_test_df = process_clean_data(raw_test_path)

# Step 5: Standardize and apply PCA
test_scaled = scaler.transform(cleaned_test_df)
test_reduced = pca.transform(test_scaled)

# Step 6: Predict
predictions = mlp_model.predict(test_reduced)
predicted_labels = label_encoder.inverse_transform(predictions)

# Step 7: Output predictions
print("🔮 Predicted Food Labels:")
for i, label in enumerate(predicted_labels):
    print(f"Sample {i + 1}: {label}")

# Step 8: Evaluate
if true_labels is not None:
    true_encoded = label_encoder.transform(true_labels)
    acc = accuracy_score(true_encoded, predictions)
    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(true_encoded, predictions, target_names=label_encoder.classes_))

    # Confusion matrix
    cm = confusion_matrix(true_encoded, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix (Test Set)")
    plt.show()
else:
    print("\nℹ️ No ground truth labels found in test dataset — skipping evaluation.")
