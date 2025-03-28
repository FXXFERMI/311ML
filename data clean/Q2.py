import re
import sys
import csv
import random
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 1650)  # Adjust the number as needed

#### Import Data ####
# Load the dataset
data_path = './cleaned_data_combined.csv'
df = pd.read_csv(data_path)

clean_data_path = './clean data/Q2_clean.csv'
q2 = 'Q2: How many ingredients would you expect this food item to contain?'

    
def convert_numbers(text):
    words = []
    for token in text.split():
        # Handle number ranges like "7 to 8" or "7-8"
        if "to" in token or "-" in token:
            range_values = [int(val) for val in re.split(r"[-\s]", token) if val.isdigit()]
            if len(range_values) == 2:
                words.append(str(int(np.mean(range_values))))  # Convert range to its mean

        elif token.isdigit():
            words.append(token)  # Keep numbers as words
        else:
            words.append(token)  # Keep normal words

    return " ".join(words) if words else "unknown"

df[q2] = df[q2].astype(str).apply(convert_numbers)

df[q2] = (
    df[q2]
    .astype(str)                            # Ensure the column is string type
    .str.strip()                            # Remove leading and trailing whitespace
    .str.lower()                            # Convert to lowercase
    .str.replace("'", "", regex=False)      # Remove apostrophes
    .str.replace("\n", " ", regex=False)    # Replace newlines with space
    .str.replace("-", " ", regex=False)     # Replace hyphens with space
    .str.replace(r"\(.*?\)", "", regex=True) # Remove anything inside parentheses
    .str.replace(r"[^a-zA-Z0-9. ]", "", regex=True)  # Keep only alphanumeric characters, decimal numbers, and spaces
    .str.replace(r"\s+", " ", regex=True)    # Replace multiple spaces with a single space
    .replace(r"^\s*$", "unknown", regex=True)  # Replace empty strings with 'unknown'
)

print(df[['id',q2]])

vocab = set()
for row in df[q2]:
    # Convert the text to lowercase and split into words
    words = row.lower().split()
    
    # Add each word to the set 
    vocab.update(words)

# convert set into list
vocab = list(vocab)

# Initialize a DataFrame to hold the word frequencies
bag_of_words_df = pd.DataFrame(columns=vocab)

label_column = "Label"

# Iterate through the dataset and count the word frequencies
for index, row in df[q2].items():
    word_counts = {word: row.split().count(word) for word in vocab}
    bag_of_words_df = bag_of_words_df._append(word_counts, ignore_index=True)

bag_of_words_df["id"] = df['id']  # or another unique identifier from your dataset
bag_of_words_df = bag_of_words_df[[ "id" ] + [col for col in bag_of_words_df.columns if col != "id"]]

bag_of_words_df[label_column] = df[label_column]

# Save the bag of words with frequencies to a CSV
bag_of_words_df.to_csv(clean_data_path, index=False)


