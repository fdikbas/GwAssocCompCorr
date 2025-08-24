"""
This script calculates compositional correlations between pairs of wells over a specified period.
The observation series is divided into non-overlapping windows of up to 3 years (36 months), and compositional
correlations are calculated separately for each window.

The script performs the following steps:
1. Read the selected well pairs from a CSV file.
2. Load the observation data from another CSV file.
3. Adjust the start months to the nearest next December, March, June, or September.
4. For each pair of wells, divide the observation series into non-overlapping 3-year windows.
5. For each window, generate compositions of the data and calculate correlations.
6. Calculate additional statistics (mean, standard deviation, median, IQR) of the obtained correlations.
7. Write the results, including the start and end of each evaluated window (year and month), to an output CSV file.
8. Generate another output file listing descriptive statistics between the whole series of each of the compared wells regardless of the windows.

The output CSV files contain the following columns:
- Best.and.Worst.Correlated.Compositions.csv:
  - Well1
  - Well2
  - MaxCorr
  - PearsonCorr
  - MinCorr
  - CorrDiff
  - HighestCorrComp
  - LowestCorrComp
  - MeanCorr
  - StdCorr
  - MedianCorr
  - IQRCorr
  - WindowStart
  - WindowEnd
  - Observations
- Descriptive.Statistics.of.Compared.Well.Series.csv:
  - Well1
  - Well2
  - StartMonth
  - EndMonth
  - Duration
  - MeanWell1
  - StdWell1
  - MinWell1
  - MaxWell1
  - MeanWell2
  - StdWell2
  - MinWell2
  - MaxWell2
  - PearsonCorr
"""

import pandas as pd
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_compositions(n, min_part_size, max_parts):
    """
    Generates all compositions of n where each part is at least min_part_size
    and the number of parts is limited by max_parts.
    """
    def helper(n, min_part_size, current):
        if n == 0 and len(current) <= max_parts:
            yield current
        elif len(current) < max_parts and n >= min_part_size:
            for i in range(min_part_size, n + 1):
                yield from helper(n - i, min_part_size, current + [i])

    return list(helper(n, min_part_size, []))

def adjust_start_month(month):
    """
    Adjust the start month to the nearest next December, March, June, or September.
    """
    if month in [1, 2]:
        return 3
    elif month in [4, 5]:
        return 6
    elif month in [7, 8]:
        return 9
    elif month in [10, 11]:
        return 12
    else:
        return month

def calculate_compositional_correlations(well1, well2, start_month, end_month, num_observations, df, output_file):
    """
    Calculates the compositional correlations between two wells over the specified period.
    Divides the observation series into non-overlapping windows of up to 3 years (36 months) and calculates
    compositional correlations separately for each window.
    """
    print(f"Calculating association between {well1} and {well2}")
    print(f"Common duration (n): {num_observations}")

    max_parts = 12
    min_part_size = 3  # Minimum part size for division

    # Adjust the start month
    start_month = start_month.replace(month=adjust_start_month(start_month.month))
    
    # Divide the time series into 3-year windows
    common_periods = df[(df['STATION'] == well1) | (df['STATION'] == well2)]
    common_periods = common_periods[common_periods['MSMT_DATE'].between(start_month, end_month)]
    common_periods = common_periods.pivot(index='MSMT_DATE', columns='STATION', values='WSE_imputed')
    common_periods = common_periods.dropna()

    if common_periods.empty:
        print(f"No common periods found for {well1} and {well2} between {start_month} and {end_month}")
        return

    DataSeries1 = common_periods[well1].tolist()
    DataSeries2 = common_periods[well2].tolist()
    dates = common_periods.index.tolist()

    window_size = 36  # 3 years = 36 months
    num_windows = max(1, len(dates) // window_size)  # Ensure at least one window

    for window_start_idx in range(0, num_windows * window_size, window_size):
        window_data1 = DataSeries1[window_start_idx:window_start_idx + window_size]
        window_data2 = DataSeries2[window_start_idx:window_start_idx + window_size]
        n = len(window_data1)

        if n < min_part_size:
            continue  # Skip if window is too small

        if n >= 36:
            m = max(min_part_size, n // max_parts)  # Calculate m as floor(n/12) if n is high, otherwise it is 3
        else:
            m = min_part_size

        window_start = dates[window_start_idx]
        window_end = dates[min(window_start_idx + window_size - 1, len(dates) - 1)]  # Handle case when n < 36

        MaxCorr = -1.0
        MinCorr = 1.0
        HighestCorrelatedComposition = []
        LowestCorrelatedComposition = []
        all_correlations = []

        compositions = generate_compositions(n, m, max_parts)
        total_compositions = len(compositions)
        print(f"Evaluating {well1} and {well2} in window starting {window_start.strftime('%Y-%m')}")

        for composition in compositions:
            ElementNumber = 0
            MeansOfCompositions1 = []
            MeansOfCompositions2 = []

            for part_size in composition:
                Sum1 = sum(window_data1[ElementNumber:ElementNumber + part_size])
                Sum2 = sum(window_data2[ElementNumber:ElementNumber + part_size])
                MeansOfCompositions1.append(Sum1 / part_size)
                MeansOfCompositions2.append(Sum2 / part_size)
                ElementNumber += part_size

            ElementNumber = 0
            TotalMultipliedDifferences = 0.0
            TotalDifferenceSquares1 = 0.0
            TotalDifferenceSquares2 = 0.0
            IndexOfi = 0

            for part_size in composition:
                for j in range(part_size):
                    Difference1 = window_data1[ElementNumber] - MeansOfCompositions1[IndexOfi]
                    Difference2 = window_data2[ElementNumber] - MeansOfCompositions2[IndexOfi]
                    TotalMultipliedDifferences += Difference1 * Difference2
                    TotalDifferenceSquares1 += Difference1 * Difference1
                    TotalDifferenceSquares2 += Difference2 * Difference2
                    ElementNumber += 1
                IndexOfi += 1

            Denominator = math.sqrt(TotalDifferenceSquares1 * TotalDifferenceSquares2)
            if Denominator == 0:
                Correlation = 99
            else:
                Correlation = TotalMultipliedDifferences / Denominator

            all_correlations.append(Correlation)

            if Correlation > MaxCorr:
                MaxCorr = Correlation
                HighestCorrelatedComposition = composition
            if Correlation < MinCorr:
                MinCorr = Correlation
                LowestCorrelatedComposition = composition

        PearsonCorrelation = pd.Series(window_data1).corr(pd.Series(window_data2))

        # Calculate additional statistics
        MeanCorr = np.mean(all_correlations)
        StdCorr = np.std(all_correlations)
        MedianCorr = np.median(all_correlations)
        IQRCorr = np.percentile(all_correlations, 75) - np.percentile(all_correlations, 25)

        with open(output_file, "a") as f:
            f.write(",".join(map(str, (well1, well2, MaxCorr, PearsonCorrelation, MinCorr, MaxCorr - MinCorr, 
                                       HighestCorrelatedComposition, LowestCorrelatedComposition, MeanCorr, StdCorr, 
                                       MedianCorr, IQRCorr, window_start.strftime('%Y-%m'), window_end.strftime('%Y-%m'), len(window_data1)))) + "\n")

def calculate_descriptive_statistics(well1, well2, start_month, end_month, df, descriptive_statistics_file):
    """
    Calculates descriptive statistics between the whole series of each of the compared wells regardless of the windows.
    """
    print(f"Calculating descriptive statistics between {well1} and {well2}")

    # Filter the data for the specified wells and period
    common_periods = df[(df['STATION'] == well1) | (df['STATION'] == well2)]
    common_periods = common_periods[common_periods['MSMT_DATE'].between(start_month, end_month)]
    common_periods = common_periods.pivot(index='MSMT_DATE', columns='STATION', values='WSE_imputed')
    common_periods = common_periods.dropna()

    if common_periods.empty:
        print(f"No common periods found for {well1} and {well2} between {start_month} and {end_month}")
        return

    DataSeries1 = common_periods[well1]
    DataSeries2 = common_periods[well2]

    # Calculate descriptive statistics
    start_date = common_periods.index.min().strftime('%Y-%m')
    end_date = common_periods.index.max().strftime('%Y-%m')
    duration = len(common_periods)
    
    MeanWell1 = DataSeries1.mean()
    StdWell1 = DataSeries1.std()
    MinWell1 = DataSeries1.min()
    MaxWell1 = DataSeries1.max()

    MeanWell2 = DataSeries2.mean()
    StdWell2 = DataSeries2.std()
    MinWell2 = DataSeries2.min()
    MaxWell2 = DataSeries2.max()

    PearsonCorr = DataSeries1.corr(DataSeries2)

    with open(descriptive_statistics_file, "a") as f:
        f.write(",".join(map(str, (well1, well2, start_date, end_date, duration, MeanWell1, StdWell1, MinWell1, MaxWell1, 
                                   MeanWell2, StdWell2, MinWell2, MaxWell2, PearsonCorr))) + "\n")

def read_selected_well_pairs(file_path):
    """
    Reads the selected well pairs from the specified CSV file.
    """
    print(f"Reading selected well pairs from {file_path}...")

    # Read the CSV file
    selected_well_pairs = pd.read_csv(file_path)

    # Convert Start Month and End Month to datetime
    selected_well_pairs["Start Month"] = pd.to_datetime(selected_well_pairs["Start Month"], format='%Y-%m', errors='coerce')
    selected_well_pairs["End Month"] = pd.to_datetime(selected_well_pairs["End Month"], format='%Y-%m', errors='coerce')

    # Adjust the start months
    selected_well_pairs["Start Month"] = selected_well_pairs["Start Month"].apply(lambda date: date.replace(month=adjust_start_month(date.month)))

    # Print each variable read
    for index, row in selected_well_pairs.iterrows():
        print(f"Row {index+1}: Well1={row['Well1']}, Well2={row['Well2']}, Start Month={row['Start Month']}, End Month={row['End Month']}, Common Duration={row['Common Duration']}")

    return selected_well_pairs

def main():
    """
    Main function to read input files, process each well pair, and calculate compositional correlations.
    """
    selected_well_pairs_file = "Well.Pairs.for.Compositional.Correlation.Calculation.Hesaplanamayanlar.[2025.03.13].v2.csv"
    observations_file = "gwl-monthly-imputed-KNN.csv"
    output_file = "Best.and.Worst.Correlated.Compositions.Hesaplanamayanlar.[2025.03.13].v2.csv"
    descriptive_statistics_file = "Descriptive.Statistics.of.Compared.Well.Series.Hesaplanamayanlar.[2025.03.13].v2.csv"

    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(observations_file, parse_dates=['MSMT_DATE'], dayfirst=True)

    # Read well pairs from the input file
    well_pairs_df = read_selected_well_pairs(selected_well_pairs_file)

    with open(output_file, "w") as f:
        f.write("Well1,Well2,MaxCorr,PearsonCorr,MinCorr,CorrDiff,HighestCorrComp,LowestCorrComp,MeanCorr,StdCorr,MedianCorr,IQRCorr,WindowStart,WindowEnd,Observations\n")

    with open(descriptive_statistics_file, "w") as f:
        f.write("Well1,Well2,StartMonth,EndMonth,Duration,MeanWell1,StdWell1,MinWell1,MaxWell1,MeanWell2,StdWell2,MinWell2,MaxWell2,PearsonCorr\n")

    print("Please wait, calculating compositional correlations...")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        future_to_pair = {executor.submit(calculate_compositional_correlations, row['Well1'], row['Well2'], row['Start Month'], row['End Month'], int(row['Common Duration']), df, output_file): (row['Well1'], row['Well2']) for index, row in well_pairs_df.iterrows()}
        for future in as_completed(future_to_pair):
            well1, well2 = future_to_pair[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{well1}, {well2} generated an exception: {exc}")

    print("Calculating descriptive statistics...")

    with ThreadPoolExecutor() as executor:
        future_to_pair = {executor.submit(calculate_descriptive_statistics, row['Well1'], row['Well2'], row['Start Month'], row['End Month'], df, descriptive_statistics_file): (row['Well1'], row['Well2']) for index, row in well_pairs_df.iterrows()}
        for future in as_completed(future_to_pair):
            well1, well2 = future_to_pair[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{well1}, {well2} generated an exception: {exc}")

    print("Processing complete.")

if __name__ == "__main__":
    main()