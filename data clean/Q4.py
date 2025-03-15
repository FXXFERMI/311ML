import sys
import csv
import random
import numpy as np
import pandas as pd

#### Import Data ####
# Load the dataset
data_path = './cleaned_data_combined.csv'
df = pd.read_csv(data_path)

clean_data_path = './clean data/Q4_clean.csv'


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

df['Q4: How much would you expect to pay for one serving of this food item?'].to_csv(clean_data_path, index=False)


