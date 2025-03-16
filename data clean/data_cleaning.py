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
col_name = 'Q1: From a scale 1 to 5, how complex is it to make this food? \
(Where 1 is the most simple, and 5 is the most complex)'
median_q1 = df[col_name].median()
df[col_name] = df[col_name].fillna(median_q1)

#####################################################################################################################
#### Q2 ####
col_name = 'Q2: How many ingredients would you expect this food item to contain?'
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
    

df[col_name] = df[col_name].str.lower()
df[col_name] = df[col_name].apply(clean_text)
df[col_name] = df[col_name].apply(get_numbers)

median_q2 = df[col_name].median()
df[col_name] = df[col_name].fillna(median_q2)


#####################################################################################################################
#### Q3 ####

# options that can be chosen
q3_choices = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner','At a party', 'Late night snack']
# this correcsponds to the number of times each option was chosen
q3_options_count = [0,0,0,0,0,0]

# split the data
q3_clean = df["Q3: In what setting would you expect this food to be served? Please check all that apply"].str.split(',')

# we are calcuating the mode so start all values at 0
for c in q3_choices:
    df[c] = 0  

for i, choice in q3_clean.items():
    # skip missing values (if choice is null and if its equal)
    if choice is not None and choice == choice:
        # convert the str to a list for easier iteration
        if isinstance(choice, str):
            choice = [choice]
        
        # split the input by the ","
        for i in range(len(choice)):
            curr_i = choice[i].split(',')
            
            # check if the current input's is in q3_choice and increase the count of that specific choice. Also, increase the overall option count
            for c in curr_i:
                if c in q3_choices:
                    df[i, c] = 1
                    if c == 'Week day lunch':
                        q3_options_count[0] += 1
                    elif c == 'Week day dinner':
                        q3_options_count[1] += 1
                    elif c == 'Weekend lunch':
                        q3_options_count[2] += 1
                    elif c == 'Weekend dinner':
                        q3_options_count[3] += 1
                    elif c == 'At a party':
                        q3_options_count[4] += 1
                    elif c == 'Late night snack':
                        q3_options_count[5] += 1

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
bow_df[label_col] = df[label_col]

# print(len(bow_df))


###########################################################################
################################ Q7 ################################
# options that can be chosen
q7_choices = ['Parents', 'Siblings', 'Friends', 'Teachers', 'Strangers']
# this correcsponds to the number of times each option was chosen
q7_options_count = [0,0,0,0,0]

# split the data
q7_clean = df["Q7: When you think about this food item, who does it remind you of?"].str.split(',')

# we are calcuating the mode so start all values at 0
for c in q7_choices:
    df[c] = 0  

for i, choice in q7_clean.items():
    # skip missing values (if choice is null and if its equal)
    if choice is not None and choice == choice:
        # convert the str to a list for easier iteration
        if isinstance(choice, str):
            choice = [choice]
        
        # split the input by the ","
        for i in range(len(choice)):
            curr_i = choice[i].split(',')
            
            # check if the current input's is in q3_choice and increase the count of that specific choice. Also, increase the overall option count
            for c in curr_i:
                if c in q7_choices:
                    df[i, c] = 1
                    if c == 'Parents':
                        q7_options_count[0] += 1
                    elif c == 'Siblings':
                        q7_options_count[1] += 1
                    elif c == 'Friends':
                        q7_options_count[2] += 1
                    elif c == 'Teachers':
                        q7_options_count[3] += 1
                    elif c == 'Strangers':
                        q7_options_count[4] += 1

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
df = df.drop("Q7: When you think about this food item, who does it remind you of?", axis=1)


###########################################################################
################################ Q8 ################################
# options that can be chosen
q8_choices = ['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)', 'I will have some of this food item with my hot sauce']
# this corresponds to the number of times each option was chosen
q8_options_count = [0,0,0,0,0]

# get the data
q8_clean = df["Q8: How much hot sauce would you add to this food item?"]

# we are calcuating the mode so start all values at 0
for c in q8_choices:
    df[c] = 0  

for i, choice in q8_clean.items():
    # skip missing values (if choice is null and if its equal)
    if choice is not None and choice == choice:
        # convert the str to a list for easier iteration
        if isinstance(choice, str):
            choice = [choice]
        
        for i in range(len(choice)):
            curr_i = choice[i]
            
            # check if the current input's is in q3_choice and increase the count of that specific choice. Also, increase the overall option count
            for c in curr_i:
                if c in q8_choices:
                    df[i, c] = 1
                    if c == 'None':
                        q8_options_count[0] += 1
                    elif c == 'A little (mild)':
                        q8_options_count[1] += 1
                    elif c == 'A moderate amount (medium)':
                        q8_options_count[2] += 1
                    elif c == 'A lot (hot)':
                        q8_options_count[3] += 1
                    elif c == 'I will have some of this food item with my hot sauce':
                        q8_options_count[4] += 1

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

print("Original DataFrame:")
print(df)

# next fill in any value that are missing based on the one-hot vector/most frequent above.
for index, row in df["Q8: How much hot sauce would you add to this food item?"].items():
    if pd.isnull(row):
        # if the row is null fill it in based on the one-hot vector
        df.at[index, 'None'] = choices_one_hot['None'] 
        df.at[index, 'A little (mild)'] = choices_one_hot['A little (mild)'] 
        df.at[index, 'A moderate amount (medium)'] = choices_one_hot['A moderate amount (medium)']
        df.at[index, 'A lot (hot)'] = choices_one_hot['A lot (hot)'] 
        df.at[index, 'I will have some of this food item with my hot sauce'] = choices_one_hot['I will have some of this food item with my hot sauce'] 

print("\nDataFrame after filling missing values:")
print(df)

# remove the data                
df = df.drop("Q8: How much hot sauce would you add to this food item?", axis=1)
