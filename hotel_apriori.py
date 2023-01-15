import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# load data:
df = pd.read_csv('hotel_booking.csv')

# check for missing values
print(df.isnull().sum())

# Replace missing values:
# agent: If no agency is given, booking was most likely made without one.
# company: If none given, it was most likely private.
# rest schould be self-explanatory.
nan_replacements = {"children:": 0.0,"country": "Unknown", "agent": 0, "company": 0}
df_clean = df.fillna(nan_replacements)

# "meal" contains values "Undefined", which is equal to SC.
df_clean["meal"].replace("Undefined", "SC", inplace=True)

# Some rows contain entreis with 0 adults, 0 children and 0 babies. 
# I'm dropping these entries with no guests.
zero_guests = list(df_clean.loc[df_clean["adults"]
                   + df_clean["children"]
                   + df_clean["babies"]==0].index)
df_clean.drop(df_clean.index[zero_guests], inplace=True)

# select only portugal rows
portugal_df = df_clean[df_clean["country"] == 'PRT']

# select columns of interest
clean_portugal_df = portugal_df[['stays_in_weekend_nights',
       'stays_in_week_nights',  'children', 'babies', 'previous_cancellations',
       'previous_bookings_not_canceled',  'booking_changes', 
       'required_car_parking_spaces', 'total_of_special_requests']]

# Defining the hot encoding function to make the data suitable
# for the concerned libraries
def hot_encode(x):
    if pd.isnull(x):
        return None
    elif x <= 0:
        return 0
    else:
        return 1

# Encoding the datasets
clean_portugal_df_encoded = clean_portugal_df.applymap(hot_encode)
# drop rows with NaN values
clean_portugal_df = clean_portugal_df.dropna()

# Building the model
frq_items = apriori(clean_portugal_df.astype(bool), min_support = 0.05, use_colnames = True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())

# Saving the results to a csv file
rules.to_csv('association_rules.csv', index=False)