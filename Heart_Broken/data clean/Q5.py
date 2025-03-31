## import packages ##
import sys
import csv
import random
import numpy as np
import pandas as pd

## import data ##
data_path = './cleaned_data_combined.csv'
df = pd.read_csv(data_path)

clean_data_path = './clean data/Q5_clean.csv'

q5 = "Q5: What movie do you think of when thinking of this food item?"

df[q5] = (
    df[q5]
    .astype(str)                            # Ensure the column is string type
    .str.strip()                            # remove leading and trailing whitespace
    .str.lower()                            # Convert to lowercase
    .str.replace("'", "", regex=False)      # Remove apostrophes
    .str.replace("\n", " ", regex=False)    # Replace newlines with space
    .str.replace("-", " ", regex=False)     # Replace hyphens with space
    .str.replace(r"\(.*?\)", "", regex=True) # Remove anything inside parentheses
    .str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)  # Keep only alphanumeric characters and spaces
    .str.replace(r"\s+", " ", regex=True)    # Replace multiple spaces with a single space
    .replace(r"^\s*$", "unknown", regex=True)  # Replace empty strings with 'without'
)

print(df[q5].head)

vocab = set()
for row in df[q5]:
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
for index, row in df[q5].items():
    word_counts = {word: row.split().count(word) for word in vocab}
    bag_of_words_df = bag_of_words_df._append(word_counts, ignore_index=True)

bag_of_words_df[label_column] = df[label_column]

# Save the bag of words with frequencies to a CSV
bag_of_words_df.to_csv(clean_data_path, index=False)
