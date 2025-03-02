### Q7 ###


# When you think about this food item, who does it remind you of?
  #Parents 
  #Siblings 
  #Friends 
  #Teachers 
  #Strangers     

# basing this on the structure of q3

import sys
import csv
import random
import numpy as np
import pandas as pd
import pickle as pk
#### Import Data ####

# Load the dataset
data_path = './cleaned_data_combined.csv'
df = pd.read_csv(data_path)

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
