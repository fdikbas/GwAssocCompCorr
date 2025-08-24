# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:44:39 2025

@author: FAT
"""

import pandas as pd
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_longest_common_period(df, well1, well2):
    """
    Finds the longest common observation period between two wells.
    """
    # Filter data for each well
    well1_data = df[df['STATION'] == well1][['MSMT_DATE', 'WSE_imputed']]
    well2_data = df[df['STATION'] == well2][['MSMT_DATE', 'WSE_imputed']]

    # Merge data on MSMT_DATE to find common periods
    common_data = pd.merge(well1_data, well2_data, on='MSMT_DATE', suffixes=('_1', '_2'))
    if common_data.shape[0] < 24:
        return None, None, 0

    # Find the longest continuous period with at least 24 observations
    common_data['MSMT_DATE'] = pd.to_datetime(common_data['MSMT_DATE'])
    common_data = common_data.sort_values(by='MSMT_DATE')
    common_data['diff'] = common_data['MSMT_DATE'].diff().dt.days.fillna(0).astype(int)

    max_period_start = None
    max_period_end = None
    max_period_length = 0
    current_period_start = None
    current_period_length = 0

    for i in range(len(common_data)):
        if i == 0 or (common_data['MSMT_DATE'].iloc[i] - common_data['MSMT_DATE'].iloc[i - 1]).days <= 31:
            if current_period_start is None:
                current_period_start = common_data['MSMT_DATE'].iloc[i]
            current_period_length += 1
        else:
            if current_period_length >= 24:
                if current_period_length > max_period_length:
                    max_period_length = current_period_length
                    max_period_start = current_period_start
                    max_period_end = common_data['MSMT_DATE'].iloc[i - 1]
            current_period_start = common_data['MSMT_DATE'].iloc[i]
            current_period_length = 1

    if current_period_length >= 24:
        if current_period_length > max_period_length:
            max_period_length = current_period_length
            max_period_start = current_period_start
            max_period_end = common_data['MSMT_DATE'].iloc[-1]

    if max_period_length >= 24:
        return max_period_start, max_period_end, max_period_length
    else:
        return None, None, 0

def process_well_pair(well_pair, df, output_file):
    well1, well2 = well_pair
    start_month, end_month, common_duration = find_longest_common_period(df, well1, well2)
    if common_duration >= 24:
        with open(output_file, "a") as f:
            f.write(f"{well1},{well2},{start_month.strftime('%Y-%m')},{end_month.strftime('%Y-%m')},{common_duration}\n")

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('gwl-monthly-imputed-KNN.csv', parse_dates=['MSMT_DATE'], dayfirst=True)

# Exclude stations with less than 24 observations
station_counts = df['STATION'].value_counts()
stations = station_counts[station_counts >= 24].index

output_file = "Well.Pairs.for.Compositional.Correlation.Calculaton.csv"
with open(output_file, "w") as f:
    f.write("Well1,Well2,Start Month,End Month,Common Duration\n")

# Initialize list of well pairs
well_pairs = list(itertools.combinations(stations, 2))

print("Please wait, processing well pairs...")

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_well_pair, well_pair, df, output_file): well_pair for well_pair in well_pairs}
    for i, future in enumerate(as_completed(futures)):
        well_pair = futures[future]
        if i % 100 == 0:
            print(f"Processed {i} well pairs")

print("Processing complete.")