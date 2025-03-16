import numpy as np
import pandas as pd
import os

# Directory where the best model weights are stored
weights_dir = './best_model_weights/'

# ✅ Step 1: Load the test dataset
test_data_path = './dataset/test.csv'
test_data = pd.read_csv(test_data_path)

# ✅ Step 2: Load stored vocab for Q5 & Q6 (from training)
vocabQ5_path = './dataset/vocabQ5.csv'
vocabQ6_path = './dataset/vocabQ6.csv'

vocabQ5 = pd.read_csv(vocabQ5_path).drop(columns=['ID']).columns.tolist()
vocabQ6 = pd.read_csv(vocabQ6_path).drop(columns=['ID']).columns.tolist()

# ✅ Step 3: Process & Normalize Numeric Features (Q1, Q2, Q4)
feature_columns = [
    'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)',
    'Q2: How many ingredients would you expect this food item to contain?',
    'Q4: How much would you expect to pay for one serving of this food item?'
]

# Load training data to extract mean, std, and median for normalization
train_data_path = './dataset/train_dataset.csv'
train_data = pd.read_csv(train_data_path)

# Extract mean, std, and median for numeric columns
median_values = train_data[feature_columns].median()
mean_values = train_data[feature_columns].mean()
std_values = train_data[feature_columns].std()

# ✅ Function to Extract Numbers from Strings (Used for Q2 & Q4)
def extract_numbers(text):
    if pd.isna(text):
        return np.nan
    text = str(text).lower()
    numbers = []
    for word in text.split():
        if '-' in word or 'to' in word:
            values = [int(x) for x in word.replace('to', '-').split('-') if x.isdigit()]
            if len(values) == 2:
                numbers.append(np.mean(values))
        elif word.isdigit():
            numbers.append(int(word))
    return np.nan if not numbers else np.mean(numbers)

# ✅ Ensure Numeric Processing and Apply Normalization
for col in feature_columns:
    if col in ['Q2: How many ingredients would you expect this food item to contain?', 
               'Q4: How much would you expect to pay for one serving of this food item?']:
        test_data[col] = test_data[col].astype(str).apply(extract_numbers)

    test_data[col] = test_data[col].fillna(median_values[col])  # Fill missing values with median
    test_data[col] = (test_data[col] - mean_values[col]) / std_values[col]  # Z-score normalization

# ✅ Step 4: One-Hot Encode Categorical Features (Q3, Q7, Q8)
q3_choices = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
q7_choices = ['Parents', 'Siblings', 'Friends', 'Teachers', 'Strangers']
q8_choices = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)', 'I will have some of this food item with my hot sauce']

# Initialize new columns with zeros
for c in q3_choices + q7_choices + q8_choices:
    test_data[c] = 0

# Process Q3 (Food serving settings)
q3_clean = test_data["Q3: In what setting would you expect this food to be served? Please check all that apply"].fillna('').str.split(',')
for i, choices in q3_clean.items():
    for c in choices:
        c = c.strip()
        if c in q3_choices:
            test_data.at[i, c] = 1

# Process Q7 (Who does this food remind you of?)
q7_clean = test_data["Q7: When you think about this food item, who does it remind you of?"].fillna('').str.split(',')
for i, choices in q7_clean.items():
    for c in choices:
        c = c.strip()
        if c in q7_choices:
            test_data.at[i, c] = 1

# Process Q8 (Hot sauce preference)
q8_clean = test_data["Q8: How much hot sauce would you add to this food item?"].fillna('')
for i, choice in q8_clean.items():
    if choice in q8_choices:
        test_data.at[i, choice] = 1
    else:
        test_data.at[i, 'None'] = 1  # Default category for missing values

# ✅ Step 5: Process BoW Representation for Q5 (Movies)
q5 = "Q5: What movie do you think of when thinking of this food item?"
test_data[q5] = test_data[q5].fillna("unknown").str.lower().str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)

# Initialize test BoW for Q5
bow_q5_test = pd.DataFrame(np.zeros((len(test_data), len(vocabQ5)), dtype=int), columns=vocabQ5)

# Encode words from test data
for i, row in test_data[q5].items():
    words = row.split()
    for word in words:
        if word in vocabQ5:
            bow_q5_test.at[i, word] += 1

# ✅ Step 6: Process BoW Representation for Q6 (Drinks)
q6 = "Q6: What drink would you pair with this food item?"
test_data[q6] = test_data[q6].fillna("unknown").str.lower().str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)

# Initialize test BoW for Q6
bow_q6_test = pd.DataFrame(np.zeros((len(test_data), len(vocabQ6)), dtype=int), columns=vocabQ6)

# Encode words from test data
for i, row in test_data[q6].items():
    words = row.split()
    for word in words:
        if word in vocabQ6:
            bow_q6_test.at[i, word] += 1

# ✅ Step 7: Merge BoW and One-Hot Features with Test Data
X_test = np.hstack((
    test_data[feature_columns + q3_choices + q7_choices + q8_choices].values, 
    bow_q5_test.values, 
    bow_q6_test.values
))

# ✅ Step 8: Load Best Model Weights
def load_weights(filename):
    """Loads CSV weight files into NumPy arrays."""
    return np.loadtxt(os.path.join(weights_dir, filename), delimiter=',')

W1 = load_weights('best_W1.csv')
b1 = load_weights('best_b1.csv')

W2 = load_weights('best_W2.csv')
b2 = load_weights('best_b2.csv')

W3 = load_weights('best_W3.csv')
b3 = load_weights('best_b3.csv')

# ✅ Step 9: Define Activation Functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ✅ Step 10: Perform Forward Propagation
Z1 = X_test @ W1 + b1  # Input to First Hidden Layer
A1 = relu(Z1)           # Apply ReLU Activation

Z2 = A1 @ W2 + b2       # Input to Second Hidden Layer
A2 = relu(Z2)           # Apply ReLU Activation

Z3 = A2 @ W3 + b3       # Input to Output Layer
probs = softmax(Z3)     # Apply Softmax for classification
predictions = np.argmax(probs, axis=1)  # Get predicted class

# ✅ Map Predicted Index to Food Names
label_mapping = sorted(train_data["Label"].unique())  # Get unique food labels
test_data['Predicted_Label'] = [label_mapping[idx] for idx in predictions]

# ✅ Save Predictions to CSV
output_path = './predictions.csv'
test_data[['id', 'Predicted_Label']].to_csv(output_path, index=False)

print(f"\n✅ Predictions saved to {output_path}!")
