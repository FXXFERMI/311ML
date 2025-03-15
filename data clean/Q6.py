## Q6 data cleaning ##
## Since the question itself is a free response, we are going to use bag of
## words to anallysis the data

## import packages ##
import sys
import csv
import random
import numpy as np
import pandas as pd

## import data ##
data_path = './cleaned_data_combined.csv'
df = pd.read_csv(data_path)
# print(len(df))
label_col = "Label"

## global values ##
clean_data_path = './clean data/Q6_clean.csv'
#####################################################
# data clean for Q6 #
#####################################################
'''
- make it all lowercase
- remove the symbols. i.e: period, apostrophes, '\n' etc.
- first replace all the missing value with 'NA', we will replace it with other
words based on our anaylsis.
'''
col_name = "Q6: What drink would you pair with this food item?"
# print(df[col_name].head(10))
df[col_name] = df[col_name].astype(str).str.lower()	
# print(df[col_name].head())


df[col_name] = df[col_name].astype(str).str.lower()	
df[col_name] = df[col_name].str.replace('\n', ' ', regex=False)
df[col_name] = df[col_name].str.replace('-', ' ', regex=False)
df[col_name] = df[col_name].str.replace(r"\(.*?\)", "", regex=True) 
df[col_name] = df[col_name].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
# print(df[col_name].head())

# df[col_name].to_csv(clean_data_path, index=False)



###### make the bow #######
# Tokenize (split into words)
tokenized_texts = [text.split() for text in df[col_name]]

# Build vocabulary (sorted for consistency)
vocab = sorted(set(word for words in tokenized_texts for word in words))

# Initialize an empty matrix (rows = documents, columns = words)
bow_matrix = np.zeros((len(df), len(vocab)), dtype=int)

# Fill the matrix with word counts
for i, words in enumerate(tokenized_texts):
    for word in words:
        bow_matrix[i, vocab.index(word)] += 1

# Convert to DataFrame (same format as CountVectorizer)
bow_df = pd.DataFrame(bow_matrix, columns=vocab)
bow_df[label_col] = df[label_col]

bow_df.to_csv(clean_data_path, index=False)
# print(len(bow_df))