import sys
import csv
import random
import numpy as np
import pandas as pd

#### Import Data ####
# Load the dataset
data_path = './cleaned_data_combined.csv'
df = pd.read_csv(data_path)

label_col = "Label"

#####################################################################################################################
#### Q1 ####
# missing values replaced with median for q1
median_q1 = df['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'].median()
df['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'] = df['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'].fillna(median_q1)

#####################################################################################################################
#### Q2 ####

def get_numbers(text):
    """
    Extract numbers from string.
    If range, calculate average.
    """
    if pd.isna(text):
        return np.nan

    text_str = str(text)

    numbers_in_text = []

    for i in text_str.split():
        # range using 'to' (ie. '7 to 8')
        if 'to' in i: 
            range_values = []
            for val in i.split('to'):
                if val.isdigit():
                    range_values.append(int(val))
            if len(range_values) == 2:
                numbers_in_text.append(int(np.mean(range_values)))

        # range using '-' (ie. '7-8')
        elif '-' in i:
            range_values = []
            for val in i.split('-'):
                if val.isdigit():
                    range_values.append(int(val))
            if len(range_values) == 2:
                numbers_in_text.append(int(np.mean(range_values)))
        elif i.isdigit():
            numbers_in_text.append(int(i))

    if len(numbers_in_text) == 0:
        return np.nan
    else:
        take_mean = np.mean(numbers_in_text)
        return int(take_mean)

def clean_text(text):
    if isinstance(text, str):
        text_str = text.replace('\n', '').replace(',', '').replace(':', '')
        return text_str
    else:
        return 'null'
    

df['Q2: How many ingredients would you expect this food item to contain?'] = df['Q2: How many ingredients would you expect this food item to contain?'].str.lower()
df['Q2: How many ingredients would you expect this food item to contain?'] = df['Q2: How many ingredients would you expect this food item to contain?'].apply(clean_text)
df['Q2: How many ingredients would you expect this food item to contain?'] = df['Q2: How many ingredients would you expect this food item to contain?'].apply(get_numbers)

median_q2 = df['Q2: How many ingredients would you expect this food item to contain?'].median()
df['Q2: How many ingredients would you expect this food item to contain?'] = df['Q2: How many ingredients would you expect this food item to contain?'].fillna(median_q2)


#####################################################################################################################
#### Q4 ####

def get_numbers(text):
    """
    Extract numbers from string.
    If range, calculate average.
    """
    if pd.isna(text):
        return np.nan

    text_str = str(text)

    numbers_in_text = []

    for i in text_str.split():
        # range using 'to' (ie. '7 to 8')
        if 'to' in i: 
            range_values = []
            for val in i.split('to'):
                if val.isdigit():
                    range_values.append(int(val))
            if len(range_values) == 2:
                numbers_in_text.append(int(np.mean(range_values)))

        # range using '-' (ie. '7-8')
        elif '-' in i:
            range_values = []
            for val in i.split('-'):
                if val.isdigit():
                    range_values.append(int(val))
            if len(range_values) == 2:
                numbers_in_text.append(int(np.mean(range_values)))
        elif i.isdigit():
            numbers_in_text.append(int(i))

    if len(numbers_in_text) == 0:
        return np.nan
    else:
        take_mean = np.mean(numbers_in_text)
        return int(take_mean)

def clean_text(text):
    if isinstance(text, str):
        text_str = text.replace('\n', '').replace(',', '').replace(':', '').replace('-', '').replace('$', '').replace('cad', '').replace('usd', '').replace('USD', '').replace('CAD', '')
        return text_str
    else:
        return 'null'
    

df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].str.lower()
df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].str.replace('\n', ' ', regex=False)
df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(clean_text)
df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(get_numbers)

median_q4 = df['Q4: How much would you expect to pay for one serving of this food item?'].median()
df['Q4: How much would you expect to pay for one serving of this food item?'] = df['Q4: How much would you expect to pay for one serving of this food item?'].fillna(median_q4)


#####################################################################################################################
#### Q5 ####
q5 = "Q5: What movie do you think of when thinking of this food item?"

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
# bag_of_words_df.to_csv(clean_data_path, index=False)

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

# print(len(bow_df))
