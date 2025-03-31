import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib  # for saving the trained model
from data_clean import process_clean_data

# Step 1: Load cleaned data
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, 'cleaned_data_combined.csv')
clean_data_path = os.path.join(base_dir, 'cleaned_train_dataset.csv')
# df = process_clean_data(csv_path)
df = pd.read_csv(clean_data_path)

# Step 2: Separate features and labels
y = df['Label']
X = df.drop(columns=['Label'])

# Step 3: Encode the labels (Pizza/Shawarma/Sushi → 0/1/2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Step 6: Train the MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

mlp.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = mlp.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 8: Save the model and encoders for later use
joblib.dump(mlp, os.path.join(base_dir, 'mlp_model.pkl'))
joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))
joblib.dump(le, os.path.join(base_dir, 'label_encoder.pkl'))

print("✅ Model, scaler, and label encoder saved successfully.")
