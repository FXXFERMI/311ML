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

cleaned_data_path = './clean data/Q7_clean.csv'

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
df = df.drop("Q7: When you think about this food item, who does it remind you of?", axis=1)

df[['Parents', 'Siblings', 'Friends', 'Teachers', 'Strangers']].to_csv(cleaned_data_path, index=False)