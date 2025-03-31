import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
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

# Step 3: Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Step 6: Define hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (128, 64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

# Step 7: Grid Search
mlp = MLPClassifier(max_iter=500, random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Step 8: Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\n📈 Best Hyperparameters:", grid_search.best_params_)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 9: Save best model and preprocessing objects
joblib.dump(best_model, os.path.join(base_dir, 'mlp_model_tuned.pkl'))
joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))
joblib.dump(le, os.path.join(base_dir, 'label_encoder.pkl'))

print("\n✅ Tuned model, scaler, and label encoder saved successfully.")
