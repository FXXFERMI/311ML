import os
import csv
import json
import math
from data_clean_wo_BoW import process_clean_data

########## Sklearn FUNTIONS #############
# === Activation functions ===
def relu(x):
    return [max(0, xi) for xi in x]

def tanh(x):
    return [math.tanh(xi) for xi in x]

def identity(x):
    return x

def logistic(x):
    return [1 / (1 + math.exp(-xi)) for xi in x]

def softmax(x):
    max_val = max(x)
    exps = [math.exp(i - max_val) for i in x]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

ACTIVATIONS = {
    "relu": relu,
    "tanh": tanh,
    "identity": identity,
    "logistic": logistic
}

# === Standardization ===
def standardize(x, mean, scale):
    return [(xi - mi) / si if si != 0 else 0 for xi, mi, si in zip(x, mean, scale)]

# === Feature Reduction ===
def reduce_dim(x, reducer):
    if reducer["type"] == "pca":
        comps = reducer["components"]
        return [sum(xi * cij for xi, cij in zip(x, col)) for col in zip(*comps)]
    elif reducer["type"] == "feature_selection":
        return [x[i] for i in reducer["top_indices"]]
    return x

# === Forward pass ===
def forward_pass(x, weights, biases, activation_name):
    activation = x
    hidden_activation = ACTIVATIONS[activation_name]

    for i in range(len(weights) - 1):
        layer_output = []
        for col in zip(*weights[i]):
            dot = sum(a * w for a, w in zip(activation, col))
            layer_output.append(dot)
        layer_output = [val + b for val, b in zip(layer_output, biases[i])]
        activation = hidden_activation(layer_output)

    final_output = []
    for col in zip(*weights[-1]):
        dot = sum(a * w for a, w in zip(activation, col))
        final_output.append(dot)
    final_output = [val + b for val, b in zip(final_output, biases[-1])]
    return softmax(final_output)

########## MAIN FUNTIONS #############
def predict_all(filename):
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'mlp_model_export_wo_BoW.json')

    with open(model_path, 'r') as f:
        model = json.load(f)

    weights = model["weights"]
    biases = model["biases"]
    scaler_mean = model["scaler"]["mean"]
    scaler_scale = model["scaler"]["scale"]
    label_encoder = model["label_encoder"]
    reducer = model["reducer"]
    activation = model["activation"]

    ###### clean test data ######
    cleaned_test_df = process_clean_data(filename)
    # print(cleaned_test_df)

    test_scaled = [standardize(row, scaler_mean, scaler_scale) for row in cleaned_test_df.values.tolist()]
    # test_reduced = test_scaled[:, top_indices]
    # Save your custom standardized version
    # with open("test_scaled_custom.txt", "w") as f1:
    #     for row in test_scaled:
    #         line = ",".join(f"{val:.6f}" for val in row)
    #         f1.write(line + "\n")
    test_reduced = [reduce_dim(row, reducer) for row in test_scaled]
    # # Save manual reduced
    # with open("test_reduced_manual.txt", "w") as f:
    #     for row in test_reduced:
    #         f.write(",".join(f"{x:.6f}" for x in row) + "\n")

    ###### predict ######
    predictions = []
    for x in test_reduced:
        output = forward_pass(x, weights, biases, activation)
        pred_label = label_encoder[output.index(max(output))]
        predictions.append(pred_label)

    return predictions

# if __name__ == '__main__':
#     test_path = os.path.join(os.path.dirname(__file__), 'test_dataset.csv')
#     preds = predict_all(test_path)
#     for i, label in enumerate(preds):
#         print(f"Sample {i+1}: {label}")