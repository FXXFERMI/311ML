### Q3 ###

# In what setting would you expect this food to be served. Check all that apply: 
  #Week day lunch 
  #Week day dinner 
  #Weekend lunch 
  #Weekend dinner 
  #At a party 
  #Late night snack

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

print("Original DataFrame:")
print(df)

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

print("\nDataFrame after filling missing values:")
print(df)

# remove the data                
df = df.drop("Q3: In what setting would you expect this food to be served? Please check all that apply", axis=1)
