import sys
import csv
import random
import numpy as np
import pandas as pd

#### Import Data ####
# Load the dataset
data_path = './cleaned_data_combined.csv'
df = pd.read_csv(data_path)

clean_data_path = './clean data/Q1_clean.csv'

#####################################################################################################################
#### Q1 ####
# missing values replaced with median for q1
median_q1 = df['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'].median()
df['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'] = df['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'].fillna(median_q1)

df['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'].to_csv(clean_data_path, index=False)

# col_name = 'Q1: From a scale 1 to 5, how complex is it to make this food? \
# (Where 1 is the most simple, and 5 is the most complex)'
# median_q1 = df[col_name].median()
# df[col_name] = df[col_name].fillna(median_q1)
