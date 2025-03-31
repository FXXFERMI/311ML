import sys
import csv
import random
import numpy as np
import pandas as pd
import re

#### Import Data ####
# Load the dataset
data_path = './dataset/train.csv'
df = pd.read_csv(data_path)

label_col = "Label" 

clean_data_path = './dataset/train_dataset.csv'
vocabQ2 = './dataset/vocabQ2.csv'
vocabQ4 = './dataset/vocabQ4.csv'
vocabQ5 = './dataset/vocabQ5.csv'
vocabQ6 = './dataset/vocabQ6.csv'


#####################################################################################################################
#### Q1 ####
# missing values replaced with median for q1
col_name = 'Q1: From a scale 1 to 5, how complex is it to make this food? \
(Where 1 is the most simple, and 5 is the most complex)'
median_q1 = df[col_name].median()
df[col_name] = df[col_name].fillna(median_q1)

# Compute mean and standard deviation
mean_q1 = df[col_name].mean()
std_q1 = df[col_name].std()

# Apply Z-score normalization manually
df[col_name] = (df[col_name] - mean_q1) / std_q1

# Print first few rows
# print(df[[col_name]].head())

# # Compute min and max
# min_q1 = df[col_name].min()
# max_q1 = df[col_name].max()

# # Apply Min-Max scaling manually
# df[col_name] = (df[col_name] - min_q1) / (max_q1 - min_q1)

# # Print first few rows
# print(df[[col_name]].head())

#####################################################################################################################
#### Q2 ####
col_name = 'Q2: How many ingredients would you expect this food item to contain?'

    
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

df[col_name] = (
    df[col_name]
    .astype(str)                            # Ensure the column is string type
    .str.strip()                            # Remove leading and trailing whitespace
    .str.lower()                            # Convert to lowercase
    .str.replace("'", "", regex=False)      # Remove apostrophes
    .str.replace("\n", " ", regex=False)    # Replace newlines with space
    .str.replace("~", " ", regex=False)     # Replace hyphens with space
    .str.replace("/", " ", regex=False)    # Replace slash with space
    .str.replace(r"\(.*?\)", "", regex=True) # Remove anything inside parentheses
    .str.replace(r"(\d) (\bto\b) (\d)", r"\1-\3", regex=True)
    .str.replace(r"[^a-zA-Z0-9.\- ]", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)    # Replace multiple spaces with a single space
    .replace(r"^\s*$", "unknown", regex=True)  # Replace empty strings with 'unknown'
)

df[col_name] = df[col_name].astype(str).apply(convert_numbers)

df[col_name] = (
    df[col_name]
    .astype(str)                            # Ensure the column is string type
    .str.strip()                            # Remove leading and trailing whitespace
    .str.lower()                            # Convert to lowercase
    .str.replace(".", "", regex=False)      # Remove apostrophes
)

vocab = set()
for row in df[col_name]:
    # Convert the text to lowercase and split into words
    words = row.lower().split()
    
    # Add each word to the set 
    vocab.update(words)

# convert set into list
vocab = list(vocab)

# Initialize a DataFrame to hold the word frequencies
bow_df_q2 = pd.DataFrame(columns=vocab)

label_column = "Label"

# Iterate through the dataset and count the word frequencies
for index, row in df[col_name].items():
    word_counts = {word: row.split().count(word) for word in vocab}
    bow_df_q2 = bow_df_q2._append(word_counts, ignore_index=True)

bow_df_q2.insert(0, 'ID', df['id'].values)  # Insert 'id' as the first column
# Print the first few rows to check the results
# print(df[[col_name]].head())


#####################################################################################################################
#### Q3 ####

# options that can be chosen
q3_choices = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner','At a party', 'Late night snack']
# this correcsponds to the number of times each option was chosen
q3_options_count = [0,0,0,0,0,0]

# split the data
q3_clean = df["Q3: In what setting would you expect this food to be served? Please check all that apply"].fillna('').str.split(',')

# we are calcuating the mode so start all values at 0
for c in q3_choices:
    df[c] = 0  

for i, choice in q3_clean.items():
    # skip missing values (if choice is null and if its equal)
    if choice is not None:
        # iterate through each choice
        for c in choice:
            c = c.strip()
            if c in q3_choices:
                df.at[i, c] = 1
                q3_options_count[q3_choices.index(c)] += 1

# next check to see if each choice is present more than 50% of the time and convert it into a vector
half_length = len(df) / 2
choices_one_hot = {}
for i in range(len(q3_choices)):
    count = q3_options_count[i]
    choice = q3_choices[i]

    # if the choice appears more than 50% of the time set the vector value to 1
    if count >= half_length:
        choices_one_hot[choice] = 1
    else: # otherwise it does not apper 50% of the time so set vector value to 0
        choices_one_hot[choice] = 0

# print("Original DataFrame:")
# print(df)

# next fill in any value that are missing based on the one-hot vector above. Loop though the q3 clumn in the data frame and check each row
for index, row in df["Q3: In what setting would you expect this food to be served? Please check all that apply"].items():
    if pd.isnull(row):
        # if the row is null fill it in based on the one-hot vector
        df.at[index, 'Week day lunch'] = choices_one_hot['Week day lunch'] 
        df.at[index, 'Week day dinner'] = choices_one_hot['Week day dinner'] 
        df.at[index, 'Weekend lunch'] = choices_one_hot['Weekend lunch']
        df.at[index, 'Weekend dinner'] = choices_one_hot['Weekend dinner'] 
        df.at[index, 'At a party'] = choices_one_hot['At a party'] 
        df.at[index, 'Late night snack'] = choices_one_hot['Late night snack'] 

#####################################################################################################################
#### Q4 ####
col_name_q4 = 'Q4: How much would you expect to pay for one serving of this food item?'

def convert_numbers(text):
    words = []
    for token in text.split():
        # Handle number ranges like "7 to 8" or "7-8"
        if "to" in token or "-" in token:
            range_values = [float(val) for val in re.split(r"[-\s]", token) if val.replace('.', '', 1).isdigit()]
            if len(range_values) == 2:
                words.append(str(int(np.mean(range_values))))  # Convert range to its mean and handle as int
        elif token.replace('.', '', 1).isdigit():
            if token.endswith(".00"):
                words.append(str(int(float(token))))  # Convert decimal to integer (cent-wise)
            else:
                words.append(str(int(float(token))))  # Convert decimal to integer (cent-wise)
        elif token.isdigit():
            words.append(token)  # Keep numbers as words
        else:
            words.append(token)  # Keep normal words

    return " ".join(words) if words else "unknown"

# df[q4] = df[q4].astype(str).apply(convert_numbers)

df[col_name_q4] = (
    df[col_name_q4]
    .astype(str)                            # Ensure the column is string type
    .str.strip()                            # Remove leading and trailing whitespace
    .str.lower()                            # Convert to lowercase
    .str.replace("'", "", regex=False)      # Remove apostrophes
    .str.replace("\n", " ", regex=False)    # Replace newlines with space
    .str.replace("/", " ", regex=False)    # Replace slash with space
    .str.replace("~", " ", regex=False)     # Replace hyphens with space
    .str.replace("cad", " ", regex=False)     # Replace cad with space
    .str.replace("usd", " ", regex=False)     # Replace usd with space
    .str.replace("dollars", " ", regex=False)     # Replace dollar with space
    .str.replace("dollar", " ", regex=False)     # Replace dollars with space
    .str.replace(r"\(.*?\)", "", regex=True) # Remove anything inside parentheses
    .str.replace(r"(\d) (\bto\b) (\d)", r"\1-\3", regex=True)
    .str.replace(r"[^a-zA-Z0-9.\- ]", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)    # Replace multiple spaces with a single space
    .replace(r"^\s*$", "unknown", regex=True)  # Replace empty strings with 'unknown'
)

df[col_name_q4] = df[col_name_q4].astype(str).apply(convert_numbers)

df[col_name_q4] = (
    df[col_name_q4]
    .astype(str)                            # Ensure the column is string type
    .str.strip()                            # Remove leading and trailing whitespace
    .str.lower()                            # Convert to lowercase
    .str.replace(".", "", regex=False)      # Remove apostrophes
)

vocab = set()
for row in df[col_name_q4]:
    # Convert the text to lowercase and split into words
    words = row.lower().split()
    
    # Add each word to the set 
    vocab.update(words)

# convert set into list
vocab = list(vocab)

# Initialize a DataFrame to hold the word frequencies
bow_df_q4 = pd.DataFrame(columns=vocab)

label_column = "Label"

# Iterate through the dataset and count the word frequencies
for index, row in df[col_name_q4].items():
    word_counts = {word: row.split().count(word) for word in vocab}
    bow_df_q4 = bow_df_q4._append(word_counts, ignore_index=True)

bow_df_q4.insert(0, 'ID', df['id'].values)  # Insert 'id' as the first column

# Print the first few rows to check the results
# print(df[['Q4: How much would you expect to pay for one serving of this food item?']].head())
#####################################################################################################################
#### Q5 ####
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

# bag_of_words_df[label_column] = df[label_column]

# Save the bag of words with frequencies to a CSV
# bag_of_words_df.to_csv(clean_data_path, index=False)
bag_of_words_df.insert(0, 'ID', df['id'].values)  # Insert 'id' as the first column




###########################################################################
################################Q6 #
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
# bow_df[label_col] = df[label_col]
# Ensure 'id' column is available in the original dataset
# print("Existing columns in bow_df:", bow_df.columns.tolist())

bow_df.insert(0, 'ID', df['id'].values)  # Insert 'id' as the first column

# print(len(bow_df))


###########################################################################
################################ Q7 ################################
# options that can be chosen
q7_choices = ['Parents', 'Siblings', 'Friends', 'Teachers', 'Strangers']
# this correcsponds to the number of times each option was chosen
q7_options_count = [0,0,0,0,0]

# split the data
q7_clean = df["Q7: When you think about this food item, who does it remind you of?"].fillna('').str.split(',')

# we are calcuating the mode so start all values at 0
for c in q7_choices:
    df[c] = 0  

for i, choice in q7_clean.items():
    # skip missing values (if choice is empty)
    if choice:
        # iterate through each choice
        for c in choice:
            c = c.strip()
            if c in q7_choices:
                df.at[i, c] = 1
                q7_options_count[q7_choices.index(c)] += 1

# next check to see if each choice is present more than 50% of the time and convert it into a vector
half_length = len(df) / 2
choices_one_hot = {}
for i in range(len(q7_choices)):
    count = q7_options_count[i]
    choice = q7_choices[i]

    # if the choice appears more than 50% of the time set the vector value to 1
    if count >= half_length:
        choices_one_hot[choice] = 1
    else: # otherwise it does not apper 50% of the time so set vector value to 0
        choices_one_hot[choice] = 0

# print("Original DataFrame:")
# print(df)

# next fill in any value that are missing based on the one-hot vector above. Loop though the q3 clumn in the data frame and check each row
for index, row in df["Q7: When you think about this food item, who does it remind you of?"].items():
    if pd.isnull(row):
        # if the row is null fill it in based on the one-hot vector
        df.at[index, 'Parents'] = choices_one_hot['Parents'] 
        df.at[index, 'Siblings'] = choices_one_hot['Siblings'] 
        df.at[index, 'Friends'] = choices_one_hot['Friends']
        df.at[index, 'Teachers'] = choices_one_hot['Teachers'] 
        df.at[index, 'Strangers'] = choices_one_hot['Strangers'] 

# print("\nDataFrame after filling missing values:")
# print(df)

# remove the data                
# df = df.drop("Q7: When you think about this food item, who does it remind you of?", axis=1)


###########################################################################
################################ Q8 ################################
# options that can be chosen
q8_choices = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)', 'I will have some of this food item with my hot sauce']
# this corresponds to the number of times each option was chosen
q8_options_count = [0,0,0,0,0]

# get the data
q8_clean = df["Q8: How much hot sauce would you add to this food item?"].fillna('')

# we are calcuating the mode so start all values at 0
for c in q8_choices:
    df[c] = 0  

# One-hot encode the choices
for i, choice in q8_clean.items():
    if choice in q8_choices:
        df.at[i, choice] = 1
        q8_options_count[q8_choices.index(choice)] += 1
    else: 
        df.at[i, 'None'] = 1
        q8_options_count[q8_choices.index('None')] += 1

# next check to see which option was the most frequent
most_freq = max(q8_options_count)
choices_one_hot = {}
for i in range(len(q8_choices)):
    count = q8_options_count[i]
    choice = q8_choices[i]

    # if the choice ais the most frequent value set the vector value to 1
    if count == most_freq:
        choices_one_hot[choice] = 1
    else: # otherwise set vector value to 0
        choices_one_hot[choice] = 0

# next fill in any value that are missing based on the one-hot vector/most frequent above.
for index, row in df["Q8: How much hot sauce would you add to this food item?"].items():
    if pd.isnull(row):
        # if the row is null fill it in based on the one-hot vector
        df.at[index, 'None'] = choices_one_hot['None'] 
        df.at[index, 'A little (mild)'] = choices_one_hot['A little (mild)'] 
        df.at[index, 'A moderate amount (medium)'] = choices_one_hot['A moderate amount (medium)']
        df.at[index, 'A lot (hot)'] = choices_one_hot['A lot (hot)'] 
        df.at[index, 'I will have some of this food item with my hot sauce'] = choices_one_hot['I will have some of this food item with my hot sauce'] 

# print("\nDataFrame after filling missing values:")
# print(df)

# remove the data                
# df = df.drop("Q8: How much hot sauce would you add to this food item?", axis=1)

# df[['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)', 'I will have some of this food item with my hot sauce']].to_csv(cleaned_data_path, index=False)

###############################################################################
################### Final combine ###################

col1 = 'Q1: From a scale 1 to 5, how complex is it to make this food? \
(Where 1 is the most simple, and 5 is the most complex)'

col31 = 'Week day lunch' 
col32 ='Week day dinner' 
col33 ='Weekend lunch' 
col34 ='Weekend dinner'
col35 ='At a party'
col36 ='Late night snack'
# col5 = "Q5: What movie do you think of when thinking of this food item?"
# col6 = "Q6: What drink would you pair with this food item?"

col71 = 'Parents'
col72 = 'Siblings'
col73 = 'Friends'
col74 = 'Teachers'
col75 = 'Strangers'

col81 ='None'
col82 ='A little (mild)'
col83 ='A moderate amount (medium)' 
col84 ='A lot (hot)' 
col85 ='I will have some of this food item with my hot sauce'


# df_selected = df[selected_columns]  # Extract only the required columns

# Combine all DataFrames row-wise
# final_combined_df = pd.concat([bag_of_words_df, bow_df, df[[col1, col2, col31,col32,col33,col34,col35,col36, col4, col71,col72,col73,col74,col75, col81, col82, col83, col84,col85, label_col]]], axis=0, ignore_index=True)

# final_combined_df.to_csv(clean_data_path, index=False)
df.rename(columns={'id': 'ID'}, inplace=True)  # Rename 'id' to 'ID'

df[['ID', col1, col31,col32,col33,col34,col35,col36, col71,col72,col73,col74,col75, col81, col82, col83, col84,col85,label_col]].to_csv(clean_data_path, index=False)
bow_df_q2.to_csv(vocabQ2, index=False)
bow_df_q4.to_csv(vocabQ4, index=False)
bag_of_words_df.to_csv(vocabQ5, index=False)
bow_df.to_csv(vocabQ6, index=False)