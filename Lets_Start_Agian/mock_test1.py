import pandas as pd
import os
from predict import predict_all
from data_clean import process_clean_data

# === Set paths ===
base_dir = os.path.dirname(__file__)

raw_test_path = os.path.join(base_dir, "test_dataset.csv")
test_data = pd.read_csv(raw_test_path)

# === Get predictions ===
predicted_labels = predict_all(raw_test_path)

# === Compute accuracy ===
true_labels = test_data['Label'].tolist()

if len(predicted_labels) != len(true_labels):
    print("❌ Mismatch in number of predictions and ground truth labels.")
else:
    correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
    accuracy = correct / len(true_labels)
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
