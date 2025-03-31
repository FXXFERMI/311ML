import sys
import csv
import random
import numpy as np
import pandas as pd
import re
import os

#### Import data ####

base_dir = os.path.dirname(__file__)
vocabQ5_path = os.path.join(base_dir, 'vocabQ5.csv')
vocabQ6_path = os.path.join(base_dir, 'vocabQ6.csv')

########################## main data clean function ############################

def process_clean_data(filename):
    
    df = pd.read_csv(filename)
    clean_data = pd.DataFrame()

    ############################################################################
    #### Q1 ####
    # missing values replaced with median for q1
    col_name = 'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'
    median_q1 = df[col_name].median()
    df[col_name] = df[col_name].fillna(median_q1)

    # Compute mean and standard deviation
    mean_q1 = df[col_name].mean()
    std_q1 = df[col_name].std()

    # Apply Z-score normalization manually
    df[col_name] = (df[col_name] - mean_q1) / std_q1
    clean_data[col_name] = df[col_name]

    ############################################################################
    #### Q2 ####
    col_name_q2 = 'Q2: How many ingredients would you expect this food item to contain?'

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
        

    df[col_name_q2] = df[col_name_q2].str.lower()
    df[col_name_q2] = df[col_name_q2].apply(clean_text)
    df[col_name_q2] = df[col_name_q2].apply(get_numbers)

    median_q2 = df[col_name_q2].median()
    df[col_name_q2] = df[col_name_q2].fillna(median_q2)

    # Compute mean and standard deviation for Z-score normalization
    mean_q2 = df[col_name_q2].mean()
    std_q2 = df[col_name_q2].std()

    # Apply Z-score normalization manually
    df[col_name_q2] = (df[col_name_q2] - mean_q2) / std_q2

    # Apply Z-score normalization manually
    clean_data = pd.concat([clean_data, df[col_name_q2]], axis=1)

    ############################################################################
    #### Q3 ####

    # options that can be chosen
    q3_choices = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 
                  'Weekend dinner','At a party', 'Late night snack']
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

    # df[['Week day lunch','Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party','Late night snack']].to_csv(vocab_test, index=False)
    clean_data[q3_choices] = df[q3_choices]
    # clean_data.to_csv(vocab_test, index=False)
    ####################################################################################################################
    #### Q4 ####
    col_name_q4 = 'Q4: How much would you expect to pay for one serving of this food item?'

    
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
        

    df[col_name_q4] = df[col_name_q4].str.lower()
    df[col_name_q4] = df[col_name_q4].str.replace('\n', ' ', regex=False)
    df[col_name_q4] = df[col_name_q4].apply(clean_text)
    df[col_name_q4] = df[col_name_q4].apply(get_numbers)

    median_q4 = df[col_name_q4].median()
    df[col_name_q4] = df[col_name_q4].fillna(median_q4)

    # Compute mean and standard deviation for Z-score normalization
    mean_q4 = df[col_name_q4].mean()
    std_q4 = df[col_name_q4].std()

    df[col_name_q4] = (df[col_name_q4] - mean_q4) / std_q4

    # Apply Z-score normalization manually
    clean_data = pd.concat([clean_data, df[col_name_q4]], axis=1)


    #####################################################################################################################
    #### Q5 ####
    q5 = "Q5: What movie do you think of when thinking of this food item?"

    vocabQ5 = list(pd.read_csv(vocabQ5_path).columns)

    df[q5] = (
        df[q5]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("'", "", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.replace("-", " ", regex=False)
        .str.replace(r"\(.*?\)", "", regex=True)
        .str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .replace(r"^\s*$", "unknown", regex=True)
    )

    # Generate BoW from vocabQ5
    bow_df_q5_test = pd.DataFrame(columns=vocabQ5)

    for index, row in df[q5].items():
        word_counts = {word: row.split().count(word) for word in vocabQ5}
        bow_df_q5_test = bow_df_q5_test._append(word_counts, ignore_index=True)

    # Append to clean_data
    clean_data = pd.concat([clean_data, bow_df_q5_test], axis=1)
    # clean_data.to_csv(vocab_test, index=False)



    ###########################################################################
    #### Q6 ####
    '''
    - make it all lowercase
    - remove the symbols. i.e: period, apostrophes, '\n' etc.
    - first replace all the missing value with 'NA', we will replace it with other
    words based on our anaylsis.
    '''
    col_name_q6 = "Q6: What drink would you pair with this food item?"

    # Load vocab from training
    vocabQ6 = list(pd.read_csv(vocabQ6_path).columns)

    # Clean text (same as training)
    df[col_name_q6] = df[col_name_q6].astype(str).str.lower()
    df[col_name_q6] = df[col_name_q6].str.replace('\n', ' ', regex=False)
    df[col_name_q6] = df[col_name_q6].str.replace('-', ' ', regex=False)
    df[col_name_q6] = df[col_name_q6].str.replace(r"\(.*?\)", "", regex=True)
    df[col_name_q6] = df[col_name_q6].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)

    # Tokenize
    tokenized_texts = [text.split() for text in df[col_name_q6]]

    # Initialize BoW matrix with zeros
    bow_matrix_q6 = np.zeros((len(df), len(vocabQ6)), dtype=int)

    # Fill BoW matrix based on vocabQ6
    for i, words in enumerate(tokenized_texts):
        for word in words:
            if word in vocabQ6:
                bow_matrix_q6[i, vocabQ6.index(word)] += 1

    # Convert to DataFrame
    bow_df_q6_test = pd.DataFrame(bow_matrix_q6, columns=vocabQ6)

    # Append to clean_data
    clean_data = pd.concat([clean_data, bow_df_q6_test], axis=1)
    # clean_data.to_csv(vocab_test, index=False)

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
    # clean_data.to_csv(vocab_test, index=False)
    clean_data[q7_choices] = df[q7_choices]
    # clean_data.to_csv(vocab_test, index=False)


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
    clean_data[q8_choices] = df[q8_choices]
    # clean_data.to_csv(vocab_test, index=False)
    ###############################################################################
    ################### Final combine ###################
    clean_data_combined = clean_data.T.groupby(level=0).sum().T
    # clean_data_combined['Label'] = df['Label']
    # print(clean_data.shape)
    # print(clean_data_combined.shape)
    
    # base_dir = os.path.dirname(__file__)
    # csv_path = os.path.join(base_dir, 'cleaned_train_dataset.csv')
    # clean_data_combined.to_csv(csv_path, index=False)
    
    return clean_data_combined


