### Q8 ###

# Q8: How much hot sauce would you add to this food item?
  #None 
  #A little (mild) 
  #A moderate amount (medium) 
  #A lot (hot) 
  #I will have some of this food item with my hot sauce 

# transform this into a one-hot vector where there is a 1 if the option was chose and a 0 otherwise.
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

print("\nDataFrame after filling missing values:")
print(df)

# remove the data                
df = df.drop("Q8: How much hot sauce would you add to this food item?", axis=1)

cleaned_data_path = 'clean data/Q8_clean.csv'
df[['None', 'A little (mild)', 'A moderate amount (medium)', 'A lot (hot)', 'I will have some of this food item with my hot sauce']].to_csv(cleaned_data_path, index=False)