import pandas as pd
import glob
import os

# Get all CSV files in the data folder
csv_files = glob.glob(os.path.join(os.path.dirname(__file__), '*.csv'))

# Read and concatenate all CSVs
all_dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

# Save to a new CSV
combined_df.to_csv('combined_potato_disease_data.csv', index=False)
print('✅ All CSV files combined into combined_potato_disease_data.csv')
